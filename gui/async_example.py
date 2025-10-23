"""Example demonstrating async simulation execution with progress tracking.

This module shows how to integrate the async execution framework
into the GraniteLedger Streamlit GUI for non-blocking simulations.
"""

import streamlit as st
import time
from typing import Any

from gui.async_executor import ProgressUpdate, JobStatus
from gui.async_simulation_wrapper import run_simulation_with_progress, get_async_runner


def example_streamlit_async_simulation():
    """Example Streamlit app demonstrating async execution.
    
    IMPORTANT: This shows the CORRECT thread-safe pattern for Streamlit.
    Do NOT modify session_state from the progress callback (worker thread).
    Instead, poll updates from the main thread using get_pending_updates().
    """
    
    st.title("Async Simulation Example")
    
    # Initialize session state for job tracking
    if 'job_id' not in st.session_state:
        st.session_state.job_id = None
    if 'job_status' not in st.session_state:
        st.session_state.job_status = None
    if 'progress_data' not in st.session_state:
        st.session_state.progress_data = {'progress': 0.0, 'message': ''}
        
    # Progress callback (runs on worker thread - do NOT touch session_state here!)
    def log_progress(update: ProgressUpdate):
        """Log progress updates (safe for worker thread)."""
        # Only log or write to files, never modify session_state
        import logging
        logging.info(f"Progress: {update.progress:.1%} - {update.message}")
        
    # Button to start simulation
    if st.button("Run Simulation", disabled=st.session_state.job_id is not None):
        # Example: replace with actual frames and configuration
        # frames = load_frames(...)
        # years = [2025, 2030, 2035]
        
        # For demonstration, we'll show the structure
        st.info("Starting async simulation...")
        
        # This is how you would start an async simulation:
        # job_id, job = run_simulation_with_progress(
        #     frames=frames,
        #     years=years,
        #     progress_callback=log_progress,  # Worker thread callback (logging only!)
        #     # Additional kwargs for run_end_to_end_from_frames
        #     use_network=True,
        #     capacity_expansion=False
        # )
        # st.session_state.job_id = job_id
        
        st.warning("Replace with actual frames and configuration")
        
    # Display progress if job is running
    if st.session_state.job_id:
        runner = get_async_runner()
        job = runner.get_job(st.session_state.job_id)
        
        if job:
            # Check current status
            status = job.check_status()
            
            # THREAD-SAFE: Poll updates from main thread
            pending_updates = job.get_pending_updates()
            for update in pending_updates:
                # Now it's safe to update session_state (we're on main thread)
                st.session_state.progress_data = {
                    'progress': update.progress,
                    'message': update.message,
                    'year': update.year,
                    'iteration': update.iteration
                }
            
            # Show progress bar and message
            progress_info = st.session_state.progress_data
            
            if status == JobStatus.RUNNING:
                st.progress(progress_info.get('progress', 0.0))
                st.text(progress_info.get('message', 'Running...'))
                
                if 'year' in progress_info and progress_info['year']:
                    st.text(f"Current year: {progress_info['year']}")
                if 'iteration' in progress_info and progress_info['iteration']:
                    st.text(f"Iteration: {progress_info['iteration']}")
                    
                # Button to cancel
                if st.button("Cancel Simulation"):
                    job.cancel()
                    st.session_state.job_id = None
                    st.rerun()
                    
                # Auto-refresh every 0.5 seconds
                time.sleep(0.5)
                st.rerun()
                
            elif status == JobStatus.COMPLETED:
                st.success("Simulation completed!")
                result = job.result
                
                # Process and display results
                # outputs = result
                # display_results(outputs)
                
                # Cleanup
                runner.cleanup_job(st.session_state.job_id)
                st.session_state.job_id = None
                
            elif status == JobStatus.FAILED:
                st.error(f"Simulation failed: {job.error}")
                runner.cleanup_job(st.session_state.job_id)
                st.session_state.job_id = None
                
            elif status == JobStatus.CANCELLED:
                st.warning("Simulation was cancelled")
                runner.cleanup_job(st.session_state.job_id)
                st.session_state.job_id = None


def integrate_into_existing_gui():
    """
    Guide for integrating async execution into the existing GUI.
    
    Steps to integrate:
    
    1. Import the async wrapper at the top of gui/app.py:
       ```python
       from gui.async_simulation_wrapper import run_simulation_with_progress
       from gui.async_executor import ProgressUpdate, JobStatus
       ```
    
    2. Add job tracking to session state initialization:
       ```python
       if 'simulation_job_id' not in st.session_state:
           st.session_state.simulation_job_id = None
       if 'simulation_progress' not in st.session_state:
           st.session_state.simulation_progress = {'progress': 0.0, 'message': ''}
       ```
    
    3. Create a progress callback function:
       ```python
       def on_simulation_progress(update: ProgressUpdate):
           st.session_state.simulation_progress = {
               'progress': update.progress,
               'message': update.message,
               'year': update.year,
               'iteration': update.iteration,
               'metadata': update.metadata
           }
       ```
    
    4. Replace synchronous execution (around line 8892 in gui/app.py):
       
       OLD (blocking):
       ```python
       outputs = runner(frames_obj, **runner_kwargs)
       ```
       
       NEW (async with thread-safe polling):
       ```python
       # Extract years from runner_kwargs
       years = runner_kwargs.get('years', [])
       
       # Start async execution
       job_id, job = run_simulation_with_progress(
           frames=frames_obj,
           years=years,
           progress_callback=on_simulation_progress,  # Logging only, no session_state!
           **runner_kwargs
       )
       
       # Store job ID in session state
       st.session_state.simulation_job_id = job_id
       
       # Poll for completion with progress updates
       progress_placeholder = st.empty()
       status_placeholder = st.empty()
       
       while not job.is_done:
           job.check_status()
           
           # THREAD-SAFE: Poll updates from main thread
           pending_updates = job.get_pending_updates()
           for update in pending_updates:
               # Safe to update session_state now (main thread)
               st.session_state.simulation_progress = {
                   'progress': update.progress,
                   'message': update.message,
                   'year': update.year,
                   'iteration': update.iteration
               }
           
           # Update UI
           progress_info = st.session_state.simulation_progress
           progress_placeholder.progress(progress_info.get('progress', 0.0))
           status_placeholder.text(progress_info.get('message', 'Running...'))
           
           # Small delay to prevent excessive redraws
           time.sleep(0.1)
           
       # Get final result
       if job.status == JobStatus.COMPLETED:
           outputs = job.result
       elif job.status == JobStatus.FAILED:
           raise job.error
       else:
           raise RuntimeError("Simulation was cancelled")
       
       # Cleanup
       get_async_runner().cleanup_job(job_id)
       st.session_state.simulation_job_id = None
       ```
    
    5. Add cancellation button in the UI:
       ```python
       if st.session_state.simulation_job_id:
           if st.button("Cancel Simulation"):
               get_async_runner().cancel_job(st.session_state.simulation_job_id)
               st.session_state.simulation_job_id = None
               st.rerun()
       ```
    
    Benefits:
    - UI remains responsive during long simulations
    - Real-time progress updates for multi-year runs
    - Ability to cancel long-running jobs
    - Better user experience with progress bars and status messages
    
    Notes:
    - The async wrapper is backward compatible with the existing engine API
    - Progress callbacks are optional; the wrapper works without them
    - Job state is managed automatically with cleanup on completion
    """
    pass


if __name__ == "__main__":
    # Run the example
    example_streamlit_async_simulation()
