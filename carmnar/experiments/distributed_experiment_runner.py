"""
Distributed Experiment Runner for MNAR Robustness Experiments
=============================================================

This module implements distributed computing infrastructure to handle
large-scale MNAR robustness experiments with n=100 replications.

Key Features:
- Parallel experiment execution across multiple cores/processes
- Load balancing and fault tolerance
- Progress tracking and checkpointing
- Resource management and monitoring
- Scalable to hundreds of experimental conditions

Author: Research Team
Date: 2025
"""

import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import signal
import os
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from queue import Empty
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentTask:
    """Represents a single experimental task."""
    task_id: str
    condition: Dict[str, Any]
    replicate_id: int
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 3600  # 1 hour default timeout
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ExperimentResult:
    """Result from a single experimental task."""
    task: ExperimentTask
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = ""


class DistributedExperimentRunner:
    """
    Distributed experiment runner with load balancing and fault tolerance.

    Handles parallel execution of experimental tasks across multiple processes,
    with automatic load balancing, checkpointing, and recovery from failures.
    """

    def __init__(self,
                 max_workers: Optional[int] = None,
                 checkpoint_interval: int = 60,
                 results_dir: str = "results/distributed_experiments",
                 memory_limit_gb: float = 8.0):
        """
        Initialize distributed runner.

        Args:
            max_workers: Maximum number of worker processes (default: CPU count)
            checkpoint_interval: Checkpoint frequency in seconds
            results_dir: Directory to save results and checkpoints
            memory_limit_gb: Memory limit per worker process
        """
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        self.checkpoint_interval = checkpoint_interval
        self.memory_limit_gb = memory_limit_gb

        # Setup directories
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # State management (lazy initialization for Windows compatibility)
        self.manager = None
        self.task_queue = None
        self.result_queue = None
        self.completed_tasks = None
        self.failed_tasks = None
        self.running_tasks = None

        # Progress tracking
        self.total_tasks = 0
        self.completed_count = None
        self.failed_count = None
        self.start_time = None

        # Control flags
        self.stop_event = None
        self.pause_event = None

        # Load balancing
        self.worker_loads = None
        self._initialized = False

        logger.info(f"Initialized distributed runner with {self.max_workers} workers")
    
    def _initialize_manager(self):
        """Lazy initialization of multiprocessing manager for Windows compatibility."""
        if not self._initialized:
            try:
                self.manager = Manager()
                self.task_queue = self.manager.Queue()
                self.result_queue = self.manager.Queue()
                self.completed_tasks = self.manager.dict()
                self.failed_tasks = self.manager.dict()
                self.running_tasks = self.manager.dict()
                self.completed_count = self.manager.Value('i', 0)
                self.failed_count = self.manager.Value('i', 0)
                self.stop_event = self.manager.Event()
                self.pause_event = self.manager.Event()
                self.worker_loads = self.manager.dict()
                for i in range(self.max_workers):
                    self.worker_loads[i] = 0
                self._initialized = True
            except Exception as e:
                # Fallback to threading-based approach for Windows
                logger.warning(f"Multiprocessing Manager failed, using threading: {e}")
                import queue
                from threading import Event
                self.task_queue = queue.Queue()
                self.result_queue = queue.Queue()
                self.completed_tasks = {}
                self.failed_tasks = {}
                self.running_tasks = {}
                self.completed_count = {'value': 0}
                self.failed_count = {'value': 0}
                self.stop_event = Event()
                self.pause_event = Event()
                self.worker_loads = {i: 0 for i in range(self.max_workers)}
                self._initialized = True

    def submit_tasks(self, tasks: List[ExperimentTask]) -> None:
        """Submit tasks to the execution queue."""
        self._initialize_manager()
        self.total_tasks = len(tasks)
        logger.info(f"Submitting {self.total_tasks} tasks for execution")

        # Sort tasks by priority (higher priority first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

        for task in sorted_tasks:
            self.task_queue.put(task)

    def run_distributed_experiments(self,
                                  experiment_function: Callable[[ExperimentTask], Dict[str, Any]],
                                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run experiments in distributed fashion.

        Args:
            experiment_function: Function to execute for each task
            progress_callback: Optional callback for progress updates

        Returns:
            Summary of execution results
        """
        self._initialize_manager()
        self.start_time = time.time()
        logger.info("Starting distributed experiment execution")

        # Start monitoring and checkpointing threads
        monitor_thread = threading.Thread(target=self._monitor_progress,
                                        args=(progress_callback,),
                                        daemon=True)
        checkpoint_thread = threading.Thread(target=self._periodic_checkpoint,
                                           daemon=True)

        monitor_thread.start()
        checkpoint_thread.start()

        try:
            # Start worker processes
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit initial batch of tasks
                futures = {}
                active_tasks = {}

                stop_check = lambda: self.stop_event.is_set() if (self.stop_event and hasattr(self.stop_event, 'is_set')) else (self.stop_event.isSet() if self.stop_event else True)
                while not stop_check():
                    # Submit new tasks if workers are available
                    while len(futures) < self.max_workers and not self.task_queue.empty():
                        try:
                            task = self.task_queue.get_nowait()
                            future = executor.submit(self._execute_task_wrapper,
                                                   experiment_function, task)
                            futures[future] = task
                            active_tasks[task.task_id] = task
                            self.running_tasks[task.task_id] = time.time()
                        except Empty:
                            break

                    # Check for completed tasks
                    for future in as_completed(futures, timeout=0.1):
                        task = futures[future]
                        del futures[future]
                        del active_tasks[task.task_id]
                        del self.running_tasks[task.task_id]

                        try:
                            result = future.result()
                            self._handle_task_completion(task, result)
                        except Exception as e:
                            self._handle_task_failure(task, str(e))

                    # Check for timed out tasks
                    current_time = time.time()
                    timed_out_tasks = []
                    for task_id, start_time in self.running_tasks.items():
                        if current_time - start_time > active_tasks[task_id].timeout:
                            timed_out_tasks.append(task_id)

                    for task_id in timed_out_tasks:
                        task = active_tasks[task_id]
                        logger.warning(f"Task {task_id} timed out, retrying...")
                        self._handle_task_failure(task, "Timeout")

                    # Exit condition
                    completed_val = self.completed_count.value if not isinstance(self.completed_count, dict) else self.completed_count['value']
                    failed_val = self.failed_count.value if not isinstance(self.failed_count, dict) else self.failed_count['value']
                    if (len(futures) == 0 and self.task_queue.empty() and
                        completed_val + failed_val >= self.total_tasks):
                        break

                    time.sleep(0.1)  # Small delay to prevent busy waiting

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down gracefully...")
            self.stop_event.set()

        finally:
            # Wait for monitoring thread
            monitor_thread.join(timeout=5)

            # Final checkpoint
            self._save_checkpoint()

        # Generate execution summary
        execution_time = time.time() - self.start_time
        summary = self._generate_execution_summary(execution_time)

        logger.info(".1f")
        logger.info(f"Completed: {summary['completed_tasks']}/{summary['total_tasks']}")
        logger.info(f"Failed: {summary['failed_tasks']}")
        logger.info(".1f")

        return summary

    def _execute_task_wrapper(self, experiment_function: Callable, task: ExperimentTask) -> ExperimentResult:
        """Wrapper for task execution with error handling and resource monitoring."""
        start_time = time.time()

        try:
            # Set process priority and memory limits
            self._configure_process_limits()

            # Execute the experiment
            result_data = experiment_function(task)

            execution_time = time.time() - start_time

            return ExperimentResult(
                task=task,
                success=True,
                result=result_data,
                execution_time=execution_time,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} failed: {str(e)}")

            return ExperimentResult(
                task=task,
                success=False,
                error=str(e),
                execution_time=execution_time,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )

    def _configure_process_limits(self) -> None:
        """Configure process resource limits."""
        try:
            # Set memory limit (soft limit)
            memory_bytes = int(self.memory_limit_gb * 1024 * 1024 * 1024)
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except ImportError:
            # resource module not available on Windows
            pass

    def _handle_task_completion(self, task: ExperimentTask, result: ExperimentResult) -> None:
        """Handle successful task completion."""
        if isinstance(self.completed_count, dict):
            self.completed_count['value'] += 1
        else:
            with self.completed_count.get_lock():
                self.completed_count.value += 1

        self.completed_tasks[task.task_id] = result.__dict__

        # Save individual result
        result_file = self.results_dir / f"result_{task.task_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result.__dict__, f, indent=2, default=str)

        logger.debug(f"Completed task {task.task_id} in {result.execution_time:.2f}s")

    def _handle_task_failure(self, task: ExperimentTask, error: str) -> None:
        """Handle task failure with retry logic."""
        task.retry_count += 1

        if task.retry_count < task.max_retries:
            logger.warning(f"Retrying task {task.task_id} (attempt {task.retry_count + 1})")
            # Add back to queue with lower priority
            task.priority -= 1
            self.task_queue.put(task)
        else:
            if isinstance(self.failed_count, dict):
                self.failed_count['value'] += 1
            else:
                with self.failed_count.get_lock():
                    self.failed_count.value += 1

            self.failed_tasks[task.task_id] = {
                'task': task.__dict__,
                'error': error,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            logger.error(f"Task {task.task_id} failed permanently after {task.max_retries} attempts: {error}")

    def _monitor_progress(self, progress_callback: Optional[Callable]) -> None:
        """Monitor execution progress and report to callback."""
        last_report = 0
        stop_check = lambda: self.stop_event.is_set() if hasattr(self.stop_event, 'is_set') else self.stop_event.isSet()

        while not stop_check():
            current_completed = self.completed_count.value if not isinstance(self.completed_count, dict) else self.completed_count['value']
            current_failed = self.failed_count.value if not isinstance(self.failed_count, dict) else self.failed_count['value']
            total_processed = current_completed + current_failed

            if total_processed > last_report:
                progress = total_processed / self.total_tasks * 100 if self.total_tasks > 0 else 0
                elapsed_time = time.time() - self.start_time if self.start_time else 0

                if progress_callback:
                    progress_callback(progress, current_completed, current_failed, elapsed_time)

                logger.info(".1f"
                           f"({current_completed} completed, {current_failed} failed)")

                last_report = total_processed

            time.sleep(5)  # Report every 5 seconds

    def _periodic_checkpoint(self) -> None:
        """Periodically save execution state."""
        stop_check = lambda: self.stop_event.is_set() if hasattr(self.stop_event, 'is_set') else self.stop_event.isSet()
        while not stop_check():
            time.sleep(self.checkpoint_interval)
            if not stop_check():
                self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        """Save current execution state to disk."""
        completed_val = self.completed_count.value if not isinstance(self.completed_count, dict) else self.completed_count['value']
        failed_val = self.failed_count.value if not isinstance(self.failed_count, dict) else self.failed_count['value']
        checkpoint = {
            'completed_tasks': dict(self.completed_tasks) if self.completed_tasks else {},
            'failed_tasks': dict(self.failed_tasks) if self.failed_tasks else {},
            'running_tasks': dict(self.running_tasks) if self.running_tasks else {},
            'completed_count': completed_val,
            'failed_count': failed_val,
            'total_tasks': self.total_tasks,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        checkpoint_file = self.results_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

    def _generate_execution_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive execution summary."""
        return {
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_count.value,
            'failed_tasks': self.failed_count.value,
            'success_rate': completed_val / self.total_tasks * 100 if self.total_tasks > 0 else 0,
            'total_execution_time': total_time,
            'average_task_time': total_time / completed_val if completed_val > 0 else 0,
            'worker_utilization': self._calculate_worker_utilization(),
            'memory_usage': self._get_memory_usage(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    def _calculate_worker_utilization(self) -> float:
        """Calculate average worker utilization."""
        if not self.worker_loads:
            return 0.0

        total_load = sum(self.worker_loads.values())
        return total_load / len(self.worker_loads) / 100.0  # Assuming load is percentage

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_gb': memory_info.rss / (1024**3),
            'vms_gb': memory_info.vms / (1024**3),
            'percent': process.memory_percent()
        }

    def load_checkpoint(self, checkpoint_file: Optional[str] = None) -> bool:
        """Load execution state from checkpoint."""
        if checkpoint_file is None:
            checkpoint_file = self.results_dir / "checkpoint.json"

        if not Path(checkpoint_file).exists():
            return False

        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)

            # Restore state
            self.completed_tasks.update(checkpoint.get('completed_tasks', {}))
            self.failed_tasks.update(checkpoint.get('failed_tasks', {}))
            self.completed_count.value = checkpoint.get('completed_count', 0)
            self.failed_count.value = checkpoint.get('failed_count', 0)
            self.total_tasks = checkpoint.get('total_tasks', 0)

            logger.info(f"Loaded checkpoint from {checkpoint_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def stop(self) -> None:
        """Stop execution gracefully."""
        logger.info("Stopping distributed experiment runner...")
        if self.stop_event:
            if hasattr(self.stop_event, 'set'):
                self.stop_event.set()
            else:
                self.stop_event.set()

        # Save final checkpoint
        if self._initialized:
            self._save_checkpoint()


def create_mnar_experiment_tasks(datasets: List[str],
                               missingness_mechanisms: List[str],
                               missingness_rates: List[float],
                               algorithms: List[str],
                               n_replicates: int = 100) -> List[ExperimentTask]:
    """
    Create experiment tasks for MNAR robustness evaluation.

    Args:
        datasets: List of dataset names
        missingness_mechanisms: List of MNAR mechanism types
        missingness_rates: List of missingness rates
        algorithms: List of algorithm names
        n_replicates: Number of replicates per condition

    Returns:
        List of ExperimentTask objects
    """
    tasks = []
    task_counter = 0

    for dataset in datasets:
        for mechanism in missingness_mechanisms:
            for rate in missingness_rates:
                for algorithm in algorithms:
                    for rep in range(n_replicates):
                        condition = {
                            'dataset': dataset,
                            'missingness_mechanism': mechanism,
                            'missingness_rate': rate,
                            'algorithm': algorithm
                        }

                        task = ExperimentTask(
                            task_id=f"task_{task_counter:06d}",
                            condition=condition,
                            replicate_id=rep,
                            priority=1,  # All tasks equal priority for now
                            timeout=1800  # 30 minutes per task
                        )

                        tasks.append(task)
                        task_counter += 1

    return tasks


def progress_callback(progress: float, completed: int, failed: int, elapsed_time: float) -> None:
    """Default progress callback function."""
    eta = (elapsed_time / max(completed + failed, 1)) * (100 - progress) if progress < 100 else 0
    print(".1f"
          ".1f")


if __name__ == "__main__":
    # Example usage
    print("Distributed Experiment Runner - Example Usage")
    print("=" * 50)

    # Create sample tasks
    tasks = create_mnar_experiment_tasks(
        datasets=['diabetes', 'heart_disease'],
        missingness_mechanisms=['sigmoid', 'gpd', 'threshold'],
        missingness_rates=[0.1, 0.3, 0.5],
        algorithms=['sm_mvpc', 'pc'],
        n_replicates=10  # Small number for demo
    )

    print(f"Created {len(tasks)} experimental tasks")

    # Initialize runner
    runner = DistributedExperimentRunner(
        max_workers=4,  # Use 4 workers for demo
        checkpoint_interval=30,
        memory_limit_gb=4.0
    )

    # Mock experiment function
    def mock_experiment(task: ExperimentTask) -> Dict[str, Any]:
        """Mock experiment function for demonstration."""
        import random
        time.sleep(random.uniform(1, 5))  # Simulate work

        # Simulate occasional failures
        if random.random() < 0.05:  # 5% failure rate
            raise Exception("Simulated experiment failure")

        return {
            'task_id': task.task_id,
            'result': 'success',
            'metrics': {
                'accuracy': random.uniform(0.7, 0.95),
                'f1_score': random.uniform(0.75, 0.90)
            }
        }

    # Run distributed experiments
    try:
        summary = runner.run_distributed_experiments(
            mock_experiment,
            progress_callback=progress_callback
        )

        print("\nExecution Summary:")
        print(f"Total tasks: {summary['total_tasks']}")
        print(f"Completed: {summary['completed_tasks']}")
        print(f"Failed: {summary['failed_tasks']}")
        print(".1f")
        print(".1f")

    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        runner.stop()
