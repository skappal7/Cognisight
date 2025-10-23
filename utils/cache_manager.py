"""
Cache Manager - Handles caching of analysis results and session management
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import polars as pl


class CacheManager:
    """
    Manages caching of analysis results and session data
    Implements smart caching to reduce LLM calls and improve performance
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Separate directories for different cache types
        self.query_cache_dir = self.cache_dir / "queries"
        self.session_cache_dir = self.cache_dir / "sessions"
        self.data_cache_dir = self.cache_dir / "data"
        
        for dir_path in [self.query_cache_dir, self.session_cache_dir, self.data_cache_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Cache metadata
        self.cache_metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {'queries': {}, 'sessions': {}, 'created': datetime.now().isoformat()}
        return {'queries': {}, 'sessions': {}, 'created': datetime.now().isoformat()}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.cache_metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def generate_cache_key(self, query: str, data_manager: Any) -> str:
        """
        Generate unique cache key based on query and data state
        
        Args:
            query: User query string
            data_manager: DataManager instance
            
        Returns:
            Unique cache key (MD5 hash)
        """
        # Create a string representing the data state
        data_state = []
        for table_name in sorted(data_manager.list_tables()):
            df = data_manager.get_table(table_name)
            # Use shape and column names as proxy for data state
            data_state.append(f"{table_name}:{df.height}x{df.width}:{','.join(df.columns)}")
        
        # Combine query and data state
        cache_input = f"{query}||{'|'.join(data_state)}"
        
        # Generate hash
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def get_from_cache(
        self,
        cache_key: str,
        max_age_hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve result from cache if available and not expired
        
        Args:
            cache_key: Cache key
            max_age_hours: Maximum age of cache entry in hours
            
        Returns:
            Cached result or None if not found/expired
        """
        cache_file = self.query_cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check age
        if cache_key in self.metadata['queries']:
            cached_time = datetime.fromisoformat(self.metadata['queries'][cache_key]['timestamp'])
            age = datetime.now() - cached_time
            
            if age > timedelta(hours=max_age_hours):
                # Cache expired
                cache_file.unlink()
                del self.metadata['queries'][cache_key]
                self._save_metadata()
                return None
        
        # Load cached result
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            
            # Update access time
            self.metadata['queries'][cache_key]['last_access'] = datetime.now().isoformat()
            self.metadata['queries'][cache_key]['access_count'] = \
                self.metadata['queries'][cache_key].get('access_count', 0) + 1
            self._save_metadata()
            
            return result
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    def save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """
        Save result to cache
        
        Args:
            cache_key: Cache key
            result: Result to cache
        """
        cache_file = self.query_cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            # Update metadata
            self.metadata['queries'][cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'last_access': datetime.now().isoformat(),
                'access_count': 0,
                'file': str(cache_file)
            }
            self._save_metadata()
            
        except Exception as e:
            print(f"Error saving to cache: {e}")
    
    def save_session(self, session_id: str, session_data: Dict[str, Any]):
        """
        Save session data
        
        Args:
            session_id: Unique session identifier
            session_data: Session data to save
        """
        session_file = self.session_cache_dir / f"{session_id}.json"
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            # Update metadata
            self.metadata['sessions'][session_id] = {
                'timestamp': datetime.now().isoformat(),
                'file': str(session_file)
            }
            self._save_metadata()
            
        except Exception as e:
            print(f"Error saving session: {e}")
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session data or None if not found
        """
        session_file = self.session_cache_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session: {e}")
            return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions"""
        sessions = []
        for session_id, info in self.metadata['sessions'].items():
            sessions.append({
                'session_id': session_id,
                'timestamp': info['timestamp'],
                'file': info['file']
            })
        return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        session_file = self.session_cache_dir / f"{session_id}.json"
        
        if session_file.exists():
            session_file.unlink()
        
        if session_id in self.metadata['sessions']:
            del self.metadata['sessions'][session_id]
            self._save_metadata()
    
    def cache_data_summary(
        self,
        table_name: str,
        summary: Dict[str, Any]
    ):
        """
        Cache data summary/profile
        
        Args:
            table_name: Name of the table
            summary: Summary data to cache
        """
        summary_file = self.data_cache_dir / f"{table_name}_summary.json"
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Error caching data summary: {e}")
    
    def get_cached_data_summary(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get cached data summary"""
        summary_file = self.data_cache_dir / f"{table_name}_summary.json"
        
        if not summary_file.exists():
            return None
        
        try:
            with open(summary_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cached summary: {e}")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        query_count = len(self.metadata['queries'])
        session_count = len(self.metadata['sessions'])
        
        # Calculate total cache size
        total_size = 0
        for file in self.cache_dir.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
        
        # Most accessed queries
        queries_with_access = [
            (key, info.get('access_count', 0))
            for key, info in self.metadata['queries'].items()
        ]
        most_accessed = sorted(queries_with_access, key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'query_cache_count': query_count,
            'session_count': session_count,
            'total_size_mb': total_size / (1024 * 1024),
            'most_accessed_queries': most_accessed,
            'created': self.metadata.get('created', 'Unknown')
        }
    
    def clear_query_cache(self):
        """Clear all query cache"""
        for file in self.query_cache_dir.glob('*.pkl'):
            file.unlink()
        
        self.metadata['queries'] = {}
        self._save_metadata()
    
    def clear_session_cache(self):
        """Clear all session cache"""
        for file in self.session_cache_dir.glob('*.json'):
            file.unlink()
        
        self.metadata['sessions'] = {}
        self._save_metadata()
    
    def clear_all(self):
        """Clear all cache"""
        self.clear_query_cache()
        self.clear_session_cache()
        
        for file in self.data_cache_dir.glob('*'):
            file.unlink()
        
        print("✅ All cache cleared")
    
    def cleanup_old_cache(self, days: int = 7):
        """
        Remove cache entries older than specified days
        
        Args:
            days: Number of days to keep cache
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Clean query cache
        queries_to_remove = []
        for cache_key, info in self.metadata['queries'].items():
            cached_time = datetime.fromisoformat(info['timestamp'])
            if cached_time < cutoff_date:
                cache_file = Path(info['file'])
                if cache_file.exists():
                    cache_file.unlink()
                queries_to_remove.append(cache_key)
        
        for key in queries_to_remove:
            del self.metadata['queries'][key]
        
        # Clean session cache
        sessions_to_remove = []
        for session_id, info in self.metadata['sessions'].items():
            session_time = datetime.fromisoformat(info['timestamp'])
            if session_time < cutoff_date:
                session_file = Path(info['file'])
                if session_file.exists():
                    session_file.unlink()
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.metadata['sessions'][session_id]
        
        self._save_metadata()
        
        print(f"✅ Cleaned {len(queries_to_remove)} query caches and {len(sessions_to_remove)} sessions")
