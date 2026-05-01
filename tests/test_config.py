import pytest
import os
from unittest.mock import patch


@pytest.mark.unit
def test_settings_defaults():
    """Test that settings load with default values."""
    with patch.dict(os.environ, {}, clear=True):
        # Force reimport to pick up new env vars
        import importlib
        import config
        importlib.reload(config)
        
        settings = config.settings
        assert settings.ENVIRONMENT == "development"
        assert settings.PORT == 8000
        assert settings.HOST == "0.0.0.0"
        assert settings.DEBUG is True


@pytest.mark.unit
def test_settings_custom_values():
    """Test that settings load custom values from environment."""
    env_vars = {
        "ENVIRONMENT": "development",
        "PORT": "9000",
        "MODEL_TEMPERATURE": "0.5",
        "CHUNK_SIZE": "1000",
    }
    
    with patch.dict(os.environ, env_vars):
        from config import Settings
        settings = Settings()
        assert settings.PORT == 9000
        assert settings.MODEL_TEMPERATURE == 0.5
        assert settings.CHUNK_SIZE == 1000


@pytest.mark.unit
def test_get_logger():
    """Test logger creation."""
    from config import get_logger
    
    logger = get_logger("test_logger")
    
    assert logger.name == "test_logger"
    assert len(logger.handlers) > 0
