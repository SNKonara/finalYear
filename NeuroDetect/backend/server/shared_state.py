"""
Shared global state for NeuroDetect Batch API.
Imported by batch_processor.py, api_utils.py and all route modules.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Mutable dicts — shared by reference across all modules
MODELS: dict = {}
PREPROCESSORS: dict = {}
MODEL_CONFIGS: dict = {}

# Paths (same layout as original batch_api.py)
BASE_DIR = Path(__file__).parent.parent.parent.parent   # C:/finalYear
PROJECT_DIR = Path(__file__).parent.parent.parent       # C:/finalYear/NeuroDetect
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results" / "batch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AUTH_USERS_COLLECTION    = 'users'
AUTH_SESSIONS_COLLECTION = 'auth_sessions'
VALID_ROLES              = {'admin', 'analyst', 'senior_analyst', 'viewer'}
