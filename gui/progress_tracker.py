"""Progress tracking for multi-year simulations.

This module provides structured progress tracking for simulations
that iterate across multiple years and/or solver iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class YearProgress:
    """Progress for a single simulation year."""
    year: int
    iterations: int = 0
    max_iterations: int = 1
    completed: bool = False
    error: Optional[str] = None
    
    @property
    def progress(self) -> float:
        """Progress fraction for this year (0.0 to 1.0)."""
        if self.completed:
            return 1.0
        if self.max_iterations > 0:
            return min(1.0, self.iterations / self.max_iterations)
        return 0.0


@dataclass
class SimulationProgress:
    """Overall simulation progress tracker."""
    years: list[int] = field(default_factory=list)
    current_year_index: int = 0
    year_progress: dict[int, YearProgress] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize year progress tracking."""
        for year in self.years:
            if year not in self.year_progress:
                self.year_progress[year] = YearProgress(year=year)
                
    @property
    def total_years(self) -> int:
        """Total number of years in simulation."""
        return len(self.years)
        
    @property
    def current_year(self) -> Optional[int]:
        """Current year being processed."""
        if 0 <= self.current_year_index < len(self.years):
            return self.years[self.current_year_index]
        return None
        
    @property
    def overall_progress(self) -> float:
        """Overall simulation progress (0.0 to 1.0)."""
        if not self.years:
            return 0.0
            
        completed_years = sum(
            1 for yp in self.year_progress.values() if yp.completed
        )
        
        if self.current_year is not None:
            current_progress = self.year_progress[self.current_year].progress
            partial_progress = current_progress / self.total_years
        else:
            partial_progress = 0.0
            
        return (completed_years + partial_progress) / max(1, self.total_years)
        
    def update_year_iteration(
        self,
        year: int,
        iteration: int,
        max_iterations: int = 1
    ) -> None:
        """Update iteration progress for a year.
        
        Args:
            year: Year being updated
            iteration: Current iteration number
            max_iterations: Maximum iterations for this year
        """
        if year not in self.year_progress:
            self.year_progress[year] = YearProgress(year=year)
            
        yp = self.year_progress[year]
        yp.iterations = iteration
        yp.max_iterations = max_iterations
        
    def complete_year(self, year: int) -> None:
        """Mark a year as completed.
        
        Args:
            year: Year to mark as complete
        """
        if year in self.year_progress:
            self.year_progress[year].completed = True
            
    def fail_year(self, year: int, error: str) -> None:
        """Mark a year as failed.
        
        Args:
            year: Year that failed
            error: Error message
        """
        if year in self.year_progress:
            yp = self.year_progress[year]
            yp.error = error
            yp.completed = False
            
    def next_year(self) -> Optional[int]:
        """Advance to next year and return it.
        
        Returns:
            Next year, or None if all years complete
        """
        self.current_year_index += 1
        return self.current_year
        
    def get_status_message(self) -> str:
        """Get human-readable status message.
        
        Returns:
            Status message describing current progress
        """
        if not self.years:
            return "No years to simulate"
            
        current = self.current_year
        if current is None:
            completed = sum(1 for yp in self.year_progress.values() if yp.completed)
            if completed == self.total_years:
                return f"Completed all {self.total_years} years"
            return "Simulation complete"
            
        yp = self.year_progress[current]
        completed_years = sum(1 for y in self.year_progress.values() if y.completed)
        
        if yp.max_iterations > 1:
            return (
                f"Year {current} - Iteration {yp.iterations}/{yp.max_iterations} "
                f"({completed_years}/{self.total_years} years complete)"
            )
        else:
            return (
                f"Processing year {current} "
                f"({completed_years}/{self.total_years} years complete)"
            )
            
    def get_detailed_status(self) -> dict[str, object]:
        """Get detailed status information.
        
        Returns:
            Dictionary with detailed progress information
        """
        return {
            'total_years': self.total_years,
            'completed_years': sum(1 for yp in self.year_progress.values() if yp.completed),
            'current_year': self.current_year,
            'overall_progress': self.overall_progress,
            'status_message': self.get_status_message(),
            'years': [
                {
                    'year': year,
                    'iterations': yp.iterations,
                    'max_iterations': yp.max_iterations,
                    'progress': yp.progress,
                    'completed': yp.completed,
                    'error': yp.error
                }
                for year, yp in sorted(self.year_progress.items())
            ]
        }
