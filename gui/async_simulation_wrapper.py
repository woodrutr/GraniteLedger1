"""Async wrapper for simulation engine with progress tracking.

This module wraps the run_end_to_end_from_frames function to provide
async execution with real-time progress updates for Streamlit GUI.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional
from collections.abc import Mapping

from gui.async_executor import AsyncSimulationJob, AsyncJobManager, ProgressUpdate
from gui.progress_tracker import SimulationProgress

try:
    from engine.run_loop import run_end_to_end_from_frames
except ImportError:
    run_end_to_end_from_frames = None

LOGGER = logging.getLogger(__name__)


class AsyncSimulationRunner:
    """Manages async simulation execution with progress tracking."""
    
    def __init__(self):
        """Initialize async simulation runner."""
        self.job_manager = AsyncJobManager()
        self._current_job_id: Optional[str] = None
        self._progress_tracker: Optional[SimulationProgress] = None
        
    def run_simulation_async(
        self,
        frames: Any,
        years: list[int],
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
        **kwargs: Any
    ) -> tuple[str, AsyncSimulationJob]:
        """Run simulation asynchronously with progress tracking.
        
        Args:
            frames: Input data frames for simulation
            years: List of simulation years
            progress_callback: Callback for progress updates
            **kwargs: Additional arguments for run_end_to_end_from_frames
            
        Returns:
            Tuple of (job_id, job) for tracking execution
        """
        if run_end_to_end_from_frames is None:
            raise ModuleNotFoundError(
                "engine.run_loop.run_end_to_end_from_frames is required"
            )
            
        import uuid
        job_id = f"sim_{uuid.uuid4().hex[:8]}"
        
        self._progress_tracker = SimulationProgress(years=years)
        
        # Will be set after job creation
        job_ref = None
        
        def wrapped_progress_callback(
            message: str,
            metadata: Mapping[str, object]
        ) -> None:
            """Internal progress callback that translates to ProgressUpdate.
            
            This chains:
            1. Engine calls this callback
            2. This updates SimulationProgress tracker
            3. This calls job.update_progress (enqueues for polling)
            4. job.update_progress calls user's progress_callback (optional)
            """
            year = metadata.get('year')
            iteration = metadata.get('iteration')
            
            if self._progress_tracker and year:
                year_int = int(year) if year else None
                iter_int = int(iteration) if iteration else None
                max_iter = int(metadata.get('max_iter', 1))
                
                if year_int:
                    self._progress_tracker.update_year_iteration(
                        year=year_int,
                        iteration=iter_int or 0,
                        max_iterations=max_iter
                    )
                    
                    if metadata.get('completed'):
                        self._progress_tracker.complete_year(year_int)
                        
                progress_value = self._progress_tracker.overall_progress
                status_msg = self._progress_tracker.get_status_message()
            else:
                progress_value = 0.0
                status_msg = message
            
            # Forward to job's update_progress to enqueue for main-thread polling
            if job_ref is not None:
                job_ref.update_progress(
                    year=int(year) if year else None,
                    iteration=int(iteration) if iteration else None,
                    message=status_msg,
                    progress=progress_value,
                    metadata=dict(metadata) if metadata else None
                )
                
        # Add wrapped callback to kwargs (will be passed to engine)
        kwargs_with_progress = kwargs.copy()
        kwargs_with_progress['progress_callback'] = wrapped_progress_callback
            
        # Create job with user's progress_callback (will be called by job.update_progress)
        job = self.job_manager.create_job(
            job_id=job_id,
            target_fn=run_end_to_end_from_frames,
            args=(frames,),
            kwargs=kwargs_with_progress,
            progress_callback=progress_callback  # User callback, called from job.update_progress
        )
        
        # Set reference so wrapped_progress_callback can call job.update_progress
        job_ref = job
        
        self._current_job_id = job_id
        job.start()
        
        return job_id, job
        
    def get_current_job(self) -> Optional[AsyncSimulationJob]:
        """Get currently running job if any."""
        if self._current_job_id:
            return self.job_manager.get_job(self._current_job_id)
        return None
        
    def cancel_current_job(self) -> bool:
        """Cancel currently running job.
        
        Returns:
            True if job was cancelled
        """
        if self._current_job_id:
            return self.job_manager.cancel_job(self._current_job_id)
        return False
        
    def cleanup_job(self, job_id: str) -> None:
        """Clean up completed job."""
        self.job_manager.cleanup_job(job_id)
        if job_id == self._current_job_id:
            self._current_job_id = None
            
    def get_progress_tracker(self) -> Optional[SimulationProgress]:
        """Get current progress tracker."""
        return self._progress_tracker


_GLOBAL_RUNNER: Optional[AsyncSimulationRunner] = None


def get_async_runner() -> AsyncSimulationRunner:
    """Get global async runner instance."""
    global _GLOBAL_RUNNER
    if _GLOBAL_RUNNER is None:
        _GLOBAL_RUNNER = AsyncSimulationRunner()
    return _GLOBAL_RUNNER


def run_simulation_with_progress(
    frames: Any,
    years: list[int],
    progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    **kwargs: Any
) -> tuple[str, AsyncSimulationJob]:
    """Convenience function to run simulation with progress tracking.
    
    Args:
        frames: Input frames for simulation
        years: List of simulation years
        progress_callback: Optional progress update callback
        **kwargs: Additional arguments for simulation
        
    Returns:
        Tuple of (job_id, job) for tracking
        
    Example:
        ```python
        def on_progress(update: ProgressUpdate):
            print(f"Progress: {update.progress:.1%} - {update.message}")
            
        job_id, job = run_simulation_with_progress(
            frames=my_frames,
            years=[2025, 2030, 2035],
            progress_callback=on_progress
        )
        
        # Poll for updates in UI loop
        while not job.is_done:
            job.check_status()
            time.sleep(0.1)
            
        result = job.result
        ```
    """
    runner = get_async_runner()
    return runner.run_simulation_async(
        frames=frames,
        years=years,
        progress_callback=progress_callback,
        **kwargs
    )
