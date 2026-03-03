"""
Firebase client for state management and real-time data streaming
Architectural Choice: Abstracted Firebase client with retry logic and connection pooling
ensures reliable state management across distributed trading instances
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError

from config import config

logger = logging.getLogger(__name__)

class FirebaseClient:
    """Firebase client with connection management and error handling"""
    
    _instance: Optional['FirebaseClient'] = None
    _client: Optional[firestore.Client] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize Firebase connection with validation"""
        try:
            # Check if Firebase app already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate({
                    "project_id": config.firebase.project_id,
                    "private_key": config.firebase.private_key,
                    "client_email": config.firebase.client_email
                })
                firebase_admin.initialize_app(cred, {
                    'projectId': config.firebase.project_id,
                    'databaseURL': config.firebase.database_url
                })
            
            self._client = firestore.client()
            logger.info("Firebase client initialized successfully")
            
        except ValueError as e:
            logger.error(f"Firebase initialization error: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected Firebase init error: {e}")
            raise
    
    @property
    def client(self) -> firestore.Client:
        """Get Firestore client with validation"""
        if self._client is None:
            raise ConnectionError("Firebase client not initialized")
        return self._client
    
    def save_trading_signal(self, 
                           symbol: str, 
                           signal_data: Dict[str, Any]) -> str:
        """
        Save trading signal to Firestore with timestamp and metadata
        
        Args:
            symbol: Trading pair symbol
            signal_data: Signal data including confidence, action, etc.
            
        Returns:
            Document ID of saved signal
        """
        try:
            collection_ref = self.client.collection("trading_signals")
            
            # Add metadata
            signal_data.update({
                "symbol": symbol,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "processed": False,
                "execution_id": f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            })
            
            doc_ref = collection_ref.add(signal_data)
            logger.info(f"Saved trading signal for {symbol}: {signal_data.get('