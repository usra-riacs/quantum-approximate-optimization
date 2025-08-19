# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.

from typing import Optional
from quapopt.circuits.backend_utilities.qiskit.qiskit_utilities import create_qiskit_session


class QiskitSessionManagerMixin:
    """
    Mixin class providing standardized Qiskit session management functionality.
    
    This mixin provides session lifecycle management for classes that need to interact
    with Qiskit backends, particularly IBM Quantum hardware. It handles both internal
    session creation and external session reuse.
    
    Features:
    - Master session management across multiple operations
    - Context manager protocol for automatic resource cleanup
    - External session support for user-managed sessions
    - Smart session ownership handling
    
    Classes using this mixin should call _init_session_management() during initialization
    and can then use start_session(), end_session(), and context manager protocols.
    """
    
    def _init_session_management(self, 
                                qiskit_backend,
                                simulation: bool = True,
                                mock_context_manager_if_simulated: bool = True,
                                session_ibm=None):
        """
        Initialize session management attributes.
        
        Args:
            qiskit_backend: Qiskit backend to use for sessions
            simulation: Whether running in simulation mode
            mock_context_manager_if_simulated: Whether to mock session for simulations
            session_ibm: External session to reuse (optional)
        """
        self._qiskit_backend = qiskit_backend
        self._simulation = simulation
        self._mock_context_manager_if_simulated = mock_context_manager_if_simulated
        self._external_session = session_ibm  # Store externally provided session
        self._session_manager = None  # Internal session manager
        self._current_session = None  # Active session

    @property
    def current_session(self):
        """Get the current session, if any."""
        return self._current_session

    def start_session(self):
        """Start a long-lived session for multiple operations."""
        if self._current_session is not None:
            raise RuntimeError("Session already active. Call end_session() first or use existing session.")
        
        if self._external_session is not None:
            # Use externally provided session (don't manage it)
            self._current_session = self._external_session
            self._session_manager = None  # We don't manage external sessions
        else:
            # Create and manage our own session
            self._session_manager = create_qiskit_session(
                qiskit_backend=self._qiskit_backend,
                mocked=self._mock_context_manager_if_simulated and self._simulation
            )
            self._current_session = self._session_manager.__enter__()

    def end_session(self):
        """End the current session (only if we created it)."""
        if self._session_manager is not None:
            # We manage this session, so clean it up
            self._session_manager.__exit__(None, None, None)
            self._session_manager = None
        # For external sessions, we don't clean up - that's the caller's responsibility
        self._current_session = None

    def __enter__(self):
        """Context manager entry - start session only if none exists."""
        if self._current_session is None:
            self.start_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - end session only if WE created it."""
        if self._session_manager is not None:
            # We created this session, so we should clean it up
            self.end_session()
        # For external sessions or already-started sessions, leave them alone
