"""Async execution framework for long-running simulations.

This module provides thread-based async execution with progress tracking
and cancellation support for energy system simulations.

Thread Safety:
- Progress updates are queued and must be consumed from the main thread
- Do NOT mutate Streamlit session state from progress callbacks
- Use a polling mechanism to pull updates from the main thread
"""

from __future__ import annotations

import threading
import queue
import traceback
from typing import Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class JobStatus(Enum):
    """Status of an async job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressUpdate:
    """Progress update from running simulation."""
    year: Optional[int] = None
    iteration: Optional[int] = None
    message: str = ""
    progress: float = 0.0  # 0.0 to 1.0
    metadata: dict[str, Any] | None = None


@dataclass
class JobResult:
    """Result from async job execution."""
    status: JobStatus
    result: Any = None
    error: Optional[Exception] = None
    error_traceback: Optional[str] = None


class AsyncSimulationJob:
    """Manages async execution of a simulation with progress tracking.
    
    Thread Safety:
    - Progress updates are placed in a queue for main-thread consumption
    - Never call Streamlit session_state methods from the progress callback
    """
    
    def __init__(
        self,
        target_fn: Callable[..., Any],
        args: tuple = (),
        kwargs: dict | None = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ):
        """Initialize async job.
        
        Args:
            target_fn: Function to execute asynchronously
            args: Positional arguments for target function
            kwargs: Keyword arguments for target function
            progress_callback: Optional callback for progress updates (runs on worker thread)
        
        Warning:
            progress_callback runs on the worker thread. Do NOT modify Streamlit
            session state directly. Use get_pending_updates() to poll from main thread.
        """
        self.target_fn = target_fn
        self.args = args
        self.kwargs = kwargs or {}
        self.progress_callback = progress_callback
        
        self._thread: Optional[threading.Thread] = None
        self._result_queue: queue.Queue[JobResult] = queue.Queue()
        self._progress_queue: queue.Queue[ProgressUpdate] = queue.Queue()
        self._cancel_event = threading.Event()
        self._status = JobStatus.PENDING
        self._result: Optional[Any] = None
        self._error: Optional[Exception] = None
        self._error_traceback: Optional[str] = None
        
    @property
    def status(self) -> JobStatus:
        """Current job status."""
        return self._status
        
    @property
    def result(self) -> Any:
        """Job result (only available when completed)."""
        return self._result
        
    @property
    def error(self) -> Optional[Exception]:
        """Job error (only available when failed)."""
        return self._error
        
    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self._status == JobStatus.RUNNING
        
    @property
    def is_done(self) -> bool:
        """Check if job is complete (success or failure)."""
        return self._status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
        
    def update_progress(
        self,
        year: Optional[int] = None,
        iteration: Optional[int] = None,
        message: str = "",
        progress: float = 0.0,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Send progress update to callback and queue.
        
        Args:
            year: Current simulation year
            iteration: Current iteration number
            message: Progress message
            progress: Progress fraction (0.0 to 1.0)
            metadata: Additional metadata
        
        Note:
            Updates are queued for main-thread consumption via get_pending_updates().
            The callback (if provided) runs on the worker thread.
        """
        update = ProgressUpdate(
            year=year,
            iteration=iteration,
            message=message,
            progress=progress,
            metadata=metadata
        )
        
        # Queue for main-thread polling
        self._progress_queue.put(update)
        
        # Also call callback if provided (runs on worker thread)
        if self.progress_callback:
            try:
                self.progress_callback(update)
            except Exception:
                pass
                
    def get_pending_updates(self) -> list[ProgressUpdate]:
        """Get all pending progress updates from the queue.
        
        Returns:
            List of progress updates (safe to call from main thread)
        """
        updates = []
        try:
            while True:
                updates.append(self._progress_queue.get_nowait())
        except queue.Empty:
            pass
        return updates
                
    def _run_with_progress(self) -> Any:
        """Execute target function with progress tracking."""
        try:
            # Only set progress_callback if not already provided
            # This allows wrapper functions to provide their own callbacks
            if 'progress_callback' in self.target_fn.__code__.co_varnames:
                if 'progress_callback' not in self.kwargs:
                    self.kwargs['progress_callback'] = self.update_progress
                
            # Only add cancel_check if the function supports it
            if 'cancel_check' in self.target_fn.__code__.co_varnames:
                self.kwargs['cancel_check'] = self._cancel_event.is_set
                
            result = self.target_fn(*self.args, **self.kwargs)
            return result
        except Exception as e:
            raise
            
    def _worker(self) -> None:
        """Worker thread function."""
        try:
            self._status = JobStatus.RUNNING
            self.update_progress(message="Starting simulation...", progress=0.0)
            
            result = self._run_with_progress()
            
            if self._cancel_event.is_set():
                self._result_queue.put(JobResult(
                    status=JobStatus.CANCELLED
                ))
            else:
                self._result_queue.put(JobResult(
                    status=JobStatus.COMPLETED,
                    result=result
                ))
        except Exception as e:
            tb = traceback.format_exc()
            self._result_queue.put(JobResult(
                status=JobStatus.FAILED,
                error=e,
                error_traceback=tb
            ))
            
    def start(self) -> None:
        """Start async execution."""
        if self._status != JobStatus.PENDING:
            raise RuntimeError(f"Cannot start job in status {self._status}")
            
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        
    def cancel(self) -> None:
        """Request cancellation of running job."""
        if self.is_running:
            self._cancel_event.set()
            self.update_progress(message="Cancelling...", progress=0.0)
            
    def check_status(self) -> JobStatus:
        """Check and update job status from result queue."""
        try:
            job_result = self._result_queue.get_nowait()
            self._status = job_result.status
            self._result = job_result.result
            self._error = job_result.error
            self._error_traceback = job_result.error_traceback
        except queue.Empty:
            pass
            
        return self._status
        
    def wait(self, timeout: Optional[float] = None) -> JobResult:
        """Wait for job completion.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            JobResult with status and result/error
        """
        if self._thread:
            self._thread.join(timeout=timeout)
            
        self.check_status()
        
        return JobResult(
            status=self._status,
            result=self._result,
            error=self._error,
            error_traceback=self._error_traceback
        )


class AsyncJobManager:
    """Manages multiple async jobs."""
    
    def __init__(self):
        """Initialize job manager."""
        self._jobs: dict[str, AsyncSimulationJob] = {}
        
    def create_job(
        self,
        job_id: str,
        target_fn: Callable[..., Any],
        args: tuple = (),
        kwargs: dict | None = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> AsyncSimulationJob:
        """Create and register a new job.
        
        Args:
            job_id: Unique job identifier
            target_fn: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            progress_callback: Progress update callback
            
        Returns:
            Created AsyncSimulationJob
        """
        job = AsyncSimulationJob(
            target_fn=target_fn,
            args=args,
            kwargs=kwargs,
            progress_callback=progress_callback
        )
        self._jobs[job_id] = job
        return job
        
    def get_job(self, job_id: str) -> Optional[AsyncSimulationJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)
        
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.
        
        Returns:
            True if job was cancelled, False if not found or not running
        """
        job = self._jobs.get(job_id)
        if job and job.is_running:
            job.cancel()
            return True
        return False
        
    def cleanup_job(self, job_id: str) -> None:
        """Remove completed job from manager."""
        self._jobs.pop(job_id, None)
        
    def active_jobs(self) -> list[str]:
        """Get list of active job IDs."""
        return [
            job_id for job_id, job in self._jobs.items()
            if not job.is_done
        ]
