"""
MongoDB Database Configuration and Operations
"""
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env.mongoDB
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.mongoDB')
load_dotenv(dotenv_path=env_path)

class MongoDB:
    """MongoDB connection and operations manager"""
    
    def __init__(self, connection_string=None, database_name=None):
        """
        Initialize MongoDB connection
        
        Args:
            connection_string: MongoDB connection string (defaults to env variable or localhost)
            database_name: Database name (defaults to 'neurodetect')
        """
        # Get connection details from environment or use defaults
        self.connection_string = connection_string or os.getenv(
            'MONGODB_URI', 
            ''
        )
        self.database_name = database_name or os.getenv('MONGODB_DATABASE', 'neurodetect')
        
        self.client = None
        self.db = None
        self.connected = False
        
        # Collection names
        self.COLLECTIONS = {
            'fraud_results': 'fraud_results',
            'transactions': 'transactions',
            'statistics': 'statistics',
            'immediate_alerts': 'immediate_alerts'
        }
    
    def connect(self):
        """Establish connection to MongoDB"""
        # Check if MongoDB is disabled
        if not self.connection_string or self.connection_string.strip()  == '':
            print("ℹ️  MongoDB disabled - results will be saved to local files only")
            self.connected = False
            return False
            
        try:
            # Create client with timeout
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get database
            self.db = self.client[self.database_name]
            self.connected = True
            
            # Create indexes for better performance
            self._create_indexes()
            
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"❌ MongoDB connection failed: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"❌ Error connecting to MongoDB: {e}")
            self.connected = False
            return False
    
    def _create_indexes(self):
        """Create indexes for better query performance"""
        try:
            # Index on transaction_id for quick lookups
            self.db[self.COLLECTIONS['fraud_results']].create_index('transaction_id')
            self.db[self.COLLECTIONS['fraud_results']].create_index('timestamp')
            self.db[self.COLLECTIONS['fraud_results']].create_index('is_fraud')
            
            self.db[self.COLLECTIONS['immediate_alerts']].create_index('transaction_id')
            self.db[self.COLLECTIONS['immediate_alerts']].create_index('timestamp')
            
            self.db[self.COLLECTIONS['transactions']].create_index('transaction_id', unique=True)
            
        except Exception:
            pass  # Indexes may already exist
    
    def insert_fraud_result(self, result):
        """
        Insert a fraud detection result
        
        Args:
            result: Dictionary containing fraud detection result
            
        Returns:
            Inserted document ID or None
        """
        if not self.connected:
            return None
        
        try:
            # Add insertion timestamp
            result['inserted_at'] = datetime.now().isoformat()
            
            # Insert into fraud_results collection
            collection = self.db[self.COLLECTIONS['fraud_results']]
            result_id = collection.insert_one(result).inserted_id
            
            return str(result_id)
            
        except Exception as e:
            print(f"❌ Error inserting fraud result: {e}")
            return None
    
    def insert_immediate_alert(self, result):
        """
        Insert an immediate fraud alert
        
        Args:
            result: Dictionary containing fraud alert data
            
        Returns:
            Inserted document ID or None
        """
        if not self.connected:
            return None
        
        try:
            # Add insertion timestamp
            result['alerted_at'] = datetime.now().isoformat()
            
            # Insert into immediate_alerts collection
            collection = self.db[self.COLLECTIONS['immediate_alerts']]
            alert_id = collection.insert_one(result).inserted_id
            
            return str(alert_id)
            
        except Exception as e:
            print(f"❌ Error inserting immediate alert: {e}")
            return None
    
    def save_statistics(self, stats):
        """
        Save detection statistics
        
        Args:
            stats: Dictionary containing statistics
            
        Returns:
            Inserted document ID or None
        """
        if not self.connected:
            return None
        
        try:
            # Add timestamp
            stats['saved_at'] = datetime.now().isoformat()
            
            # Insert into statistics collection
            collection = self.db[self.COLLECTIONS['statistics']]
            stats_id = collection.insert_one(stats).inserted_id
            
            return str(stats_id)
            
        except Exception as e:
            print(f"❌ Error saving statistics: {e}")
            return None
    
    def get_fraud_count(self, start_date=None, end_date=None):
        """
        Get count of fraud transactions
        
        Args:
            start_date: Start date filter (ISO format string)
            end_date: End date filter (ISO format string)
            
        Returns:
            Count of fraud transactions
        """
        if not self.connected:
            return 0
        
        try:
            collection = self.db[self.COLLECTIONS['fraud_results']]
            query = {'is_fraud': True}
            
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            
            return collection.count_documents(query)
            
        except Exception as e:
            print(f"❌ Error getting fraud count: {e}")
            return 0
    
    def get_recent_frauds(self, limit=10):
        """
        Get recent fraud detections
        
        Args:
            limit: Number of records to return
            
        Returns:
            List of fraud detection results
        """
        if not self.connected:
            return []
        
        try:
            collection = self.db[self.COLLECTIONS['fraud_results']]
            cursor = collection.find(
                {'is_fraud': True}
            ).sort('timestamp', -1).limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            print(f"❌ Error getting recent frauds: {e}")
            return []
    
    def get_statistics_summary(self):
        """
        Get aggregated statistics summary
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.connected:
            return {}
        
        try:
            collection = self.db[self.COLLECTIONS['fraud_results']]
            
            # Get counts
            total_count = collection.count_documents({})
            fraud_count = collection.count_documents({'is_fraud': True})
            
            # Get average processing time
            pipeline = [
                {
                    '$group': {
                        '_id': None,
                        'avg_processing_time': {'$avg': '$processing_time_ms'},
                        'avg_reconstruction_error': {'$avg': '$reconstruction_error'}
                    }
                }
            ]
            
            agg_result = list(collection.aggregate(pipeline))
            
            summary = {
                'total_transactions': total_count,
                'fraud_detected': fraud_count,
                'fraud_rate': (fraud_count / total_count * 100) if total_count > 0 else 0,
                'avg_processing_time_ms': agg_result[0]['avg_processing_time'] if agg_result else 0,
                'avg_reconstruction_error': agg_result[0]['avg_reconstruction_error'] if agg_result else 0
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error getting statistics summary: {e}")
            return {}
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.connected = False


# Singleton instance for easy import
_db_instance = None

def get_mongodb_instance(connection_string=None, database_name=None):
    """
    Get or create MongoDB singleton instance
    
    Args:
        connection_string: MongoDB connection string
        database_name: Database name
        
    Returns:
        MongoDB instance
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = MongoDB(connection_string, database_name)
        _db_instance.connect()
    
    return _db_instance
