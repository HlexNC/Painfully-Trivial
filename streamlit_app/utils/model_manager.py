"""
Model Management Utilities for Waste Sorting Assistant

Handles model downloading, caching, and version management.
"""

import os
import json
import hashlib
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple
import streamlit as st
from datetime import datetime
import shutil


class ModelManager:
    """Manages YOLOv8 model downloads and versions"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.config_file = self.models_dir / "models.json"
        self.load_config()
    
    def load_config(self):
        """Load model configuration from JSON"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "models": {},
                "default": "waste_detector_best.pt"
            }
            self.save_config()
    
    def save_config(self):
        """Save model configuration to JSON"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model"""
        return self.config["models"].get(model_name)
    
    def download_model(self, url: str, model_name: str, 
                      show_progress: bool = True) -> Tuple[bool, str]:
        """
        Download model from URL with progress tracking
        
        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            model_path = self.models_dir / model_name
            
            # Check if model already exists
            if model_path.exists():
                return True, f"Model {model_name} already exists"
            
            # Download with progress
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            if show_progress and st._is_running_with_streamlit:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            downloaded = 0
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if show_progress and st._is_running_with_streamlit:
                            progress = downloaded / total_size
                            progress_bar.progress(progress)
                            status_text.text(f"Downloading: {downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB")
            
            # Calculate checksum
            checksum = self.calculate_checksum(model_path)
            
            # Update config
            self.config["models"][model_name] = {
                "url": url,
                "path": str(model_path),
                "checksum": checksum,
                "downloaded_at": datetime.now().isoformat(),
                "size": total_size
            }
            self.save_config()
            
            if show_progress and st._is_running_with_streamlit:
                progress_bar.empty()
                status_text.empty()
            
            return True, f"Successfully downloaded {model_name}"
            
        except Exception as e:
            return False, f"Error downloading model: {str(e)}"
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def verify_model(self, model_name: str) -> bool:
        """Verify model integrity using checksum"""
        model_info = self.get_model_info(model_name)
        if not model_info:
            return False
        
        model_path = Path(model_info["path"])
        if not model_path.exists():
            return False
        
        current_checksum = self.calculate_checksum(model_path)
        return current_checksum == model_info["checksum"]
    
    def list_models(self) -> Dict[str, Dict]:
        """List all available models"""
        return self.config["models"]
    
    def delete_model(self, model_name: str) -> Tuple[bool, str]:
        """Delete a model"""
        try:
            model_info = self.get_model_info(model_name)
            if not model_info:
                return False, f"Model {model_name} not found"
            
            model_path = Path(model_info["path"])
            if model_path.exists():
                model_path.unlink()
            
            del self.config["models"][model_name]
            self.save_config()
            
            return True, f"Successfully deleted {model_name}"
        except Exception as e:
            return False, f"Error deleting model: {str(e)}"
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the path to a model file"""
        model_info = self.get_model_info(model_name)
        if model_info and Path(model_info["path"]).exists():
            return Path(model_info["path"])
        return None
    
    def set_default_model(self, model_name: str) -> bool:
        """Set the default model"""
        if model_name in self.config["models"]:
            self.config["default"] = model_name
            self.save_config()
            return True
        return False
    
    def get_default_model_path(self) -> Optional[Path]:
        """Get the path to the default model"""
        default_name = self.config.get("default")
        if default_name:
            return self.get_model_path(default_name)
        return None
    
    def backup_model(self, model_name: str, backup_dir: str = "backups") -> Tuple[bool, str]:
        """Create a backup of a model"""
        try:
            model_path = self.get_model_path(model_name)
            if not model_path:
                return False, f"Model {model_name} not found"
            
            backup_path = Path(backup_dir) / self.models_dir.name
            backup_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_path / f"{model_name.stem}_{timestamp}{model_name.suffix}"
            
            shutil.copy2(model_path, backup_file)
            
            return True, f"Backed up to {backup_file}"
        except Exception as e:
            return False, f"Backup failed: {str(e)}"


# Pre-configured model sources
AVAILABLE_MODELS = {
    "YOLOv8s - Waste Detector (Best)": {
        "url": "https://github.com/HlexNC/Painfully-Trivial/releases/download/v1.0.0/waste_detector_best.pt",
        "filename": "waste_detector_best.pt",
        "description": "Best performing model trained on 466 images, 95%+ mAP@0.5"
    },
    "YOLOv8n - Waste Detector (Fast)": {
        "url": "https://github.com/HlexNC/Painfully-Trivial/releases/download/v1.0.0/waste_detector_nano.pt",
        "filename": "waste_detector_nano.pt",
        "description": "Faster inference, suitable for edge devices, 89% mAP@0.5"
    },
    "YOLOv8m - Waste Detector (Balanced)": {
        "url": "https://github.com/HlexNC/Painfully-Trivial/releases/download/v1.0.0/waste_detector_medium.pt",
        "filename": "waste_detector_medium.pt",
        "description": "Balanced performance and speed, 93% mAP@0.5"
    }
}


def create_model_management_ui():
    """Create Streamlit UI for model management"""
    st.title("🔧 Model Management")
    
    manager = ModelManager()
    
    tab1, tab2, tab3 = st.tabs(["Download Models", "Installed Models", "Settings"])
    
    with tab1:
        st.markdown("### 📥 Available Models")
        
        for model_name, model_info in AVAILABLE_MODELS.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{model_name}**")
                st.caption(model_info["description"])
            
            with col2:
                if manager.get_model_path(model_info["filename"]):
                    st.success("✅ Installed")
                else:
                    if st.button(f"Download", key=f"dl_{model_info['filename']}"):
                        success, message = manager.download_model(
                            model_info["url"],
                            model_info["filename"]
                        )
                        if success:
                            st.success(message)
                            st.experimental_rerun()
                        else:
                            st.error(message)
    
    with tab2:
        st.markdown("### 💾 Installed Models")
        
        models = manager.list_models()
        if not models:
            st.info("No models installed yet.")
        else:
            for name, info in models.items():
                with st.expander(f"📦 {name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Size**: {info['size']/(1024*1024):.1f} MB")
                        st.markdown(f"**Downloaded**: {info['downloaded_at'][:10]}")
                        
                        if manager.verify_model(name):
                            st.success("✅ Verified")
                        else:
                            st.error("❌ Corrupted")
                    
                    with col2:
                        if st.button(f"Delete", key=f"del_{name}"):
                            success, message = manager.delete_model(name)
                            if success:
                                st.success(message)
                                st.experimental_rerun()
                            else:
                                st.error(message)
                        
                        if name != manager.config.get("default"):
                            if st.button(f"Set as Default", key=f"def_{name}"):
                                if manager.set_default_model(name):
                                    st.success(f"Set {name} as default")
                                    st.experimental_rerun()
    
    with tab3:
        st.markdown("### ⚙️ Settings")
        
        # Default model selection
        models = list(manager.list_models().keys())
        if models:
            default = st.selectbox(
                "Default Model",
                models,
                index=models.index(manager.config.get("default", models[0]))
            )
            
            if st.button("Save Settings"):
                manager.set_default_model(default)
                st.success("Settings saved!")
        
        # Model directory info
        st.markdown("### 📁 Storage Information")
        total_size = sum(info['size'] for info in manager.list_models().values())
        st.info(f"Total storage used: {total_size/(1024*1024):.1f} MB")
        st.caption(f"Models directory: {manager.models_dir.absolute()}")


if __name__ == "__main__":
    # Test the model manager
    manager = ModelManager()
    
    # Example usage
    success, message = manager.download_model(
        AVAILABLE_MODELS["YOLOv8s - Waste Detector (Best)"]["url"],
        AVAILABLE_MODELS["YOLOv8s - Waste Detector (Best)"]["filename"],
        show_progress=False
    )
    
    print(message)
    print(f"Models: {manager.list_models()}")