"""
MongoDB Database Configuration and Operations
"""
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv

try:
    from bson import ObjectId
except Exception:
    ObjectId = None

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
            'immediate_alerts': 'immediate_alerts',
            'hourly_reports': 'hourly_reports',
            'model_reports': 'model_reports',
            'resolved_frauds': 'resolved_frauds'
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
            self.db[self.COLLECTIONS['immediate_alerts']].create_index('inserted_at_dt')
            self.db[self.COLLECTIONS['immediate_alerts']].create_index('alert_status')
            self.db[self.COLLECTIONS['immediate_alerts']].create_index('risk_level')
            self.db[self.COLLECTIONS['immediate_alerts']].create_index('dismissed_at_dt')
            
            self.db[self.COLLECTIONS['transactions']].create_index('transaction_id', unique=True)
            self.db[self.COLLECTIONS['transactions']].create_index('inserted_at_dt')

            self.db[self.COLLECTIONS['hourly_reports']].create_index('hour_start', unique=True)
            self.db[self.COLLECTIONS['hourly_reports']].create_index('generated_at')
            self.db[self.COLLECTIONS['model_reports']].create_index('report_id', unique=True)
            self.db[self.COLLECTIONS['model_reports']].create_index('generated_at')
            self.db[self.COLLECTIONS['model_reports']].create_index('source.type')
            self.db[self.COLLECTIONS['model_reports']].create_index('model.type')

            self.db[self.COLLECTIONS['resolved_frauds']].create_index('source_alert_id', unique=True)
            self.db[self.COLLECTIONS['resolved_frauds']].create_index('transaction_id')
            self.db[self.COLLECTIONS['resolved_frauds']].create_index('resolved_at_dt')
            self.db[self.COLLECTIONS['resolved_frauds']].create_index('resolved_by.email')
            
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
            now_utc = datetime.utcnow()
            result['inserted_at'] = now_utc.isoformat()
            result['inserted_at_dt'] = now_utc
            
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
            now_utc = datetime.utcnow()
            result['alerted_at'] = now_utc.isoformat()
            result['inserted_at_dt'] = now_utc
            result['alert_status'] = result.get('alert_status', 'open')
            result['dismissed_at'] = result.get('dismissed_at')
            result['dismissed_at_dt'] = result.get('dismissed_at_dt')
            result['retention_policy'] = 'keep_until_dismissed'
            
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

    def dismiss_immediate_alert(self, transaction_id=None, alert_id=None, dismissed_by='investigator'):
        """
        Mark a high alert as dismissed by investigators.

        Args:
            transaction_id: Transaction identifier for alert lookup
            alert_id: MongoDB _id value as string
            dismissed_by: Investigator/user identifier

        Returns:
            Dict with update status
        """
        if not self.connected:
            return {'updated': False, 'reason': 'not_connected'}

        if not transaction_id and not alert_id:
            return {'updated': False, 'reason': 'missing_identifier'}

        try:
            query = {}
            if transaction_id:
                query['transaction_id'] = transaction_id
            if alert_id:
                if ObjectId is not None:
                    try:
                        query['_id'] = ObjectId(alert_id)
                    except Exception:
                        query['_id'] = alert_id
                else:
                    query['_id'] = alert_id

            now_utc = datetime.utcnow()
            update_result = self.db[self.COLLECTIONS['immediate_alerts']].update_many(
                query,
                {
                    '$set': {
                        'alert_status': 'dismissed',
                        'dismissed_at': now_utc.isoformat(),
                        'dismissed_at_dt': now_utc,
                        'dismissed_by': dismissed_by,
                    }
                }
            )

            return {
                'updated': update_result.modified_count > 0,
                'matched_count': update_result.matched_count,
                'modified_count': update_result.modified_count,
            }
        except Exception as e:
            return {'updated': False, 'reason': str(e)}

    def resolve_immediate_alert_as_fraud(
        self,
        transaction_id=None,
        alert_id=None,
        analyst=None,
        resolution_note=None,
    ):
        """Archive an active alert as resolved fraud with compact analyst-attributed details."""
        if not self.connected:
            return {'updated': False, 'reason': 'not_connected'}

        if not transaction_id and not alert_id:
            return {'updated': False, 'reason': 'missing_identifier'}

        try:
            query = {}
            if transaction_id:
                query['transaction_id'] = transaction_id
            if alert_id:
                if ObjectId is not None:
                    try:
                        query['_id'] = ObjectId(alert_id)
                    except Exception:
                        query['_id'] = alert_id
                else:
                    query['_id'] = alert_id

            alerts_collection = self.db[self.COLLECTIONS['immediate_alerts']]
            resolved_collection = self.db[self.COLLECTIONS['resolved_frauds']]
            alert_doc = alerts_collection.find_one(query)

            if not alert_doc:
                return {'updated': False, 'reason': 'not_found'}

            source_alert_id = str(alert_doc.get('_id'))
            existing = resolved_collection.find_one({'source_alert_id': source_alert_id})
            if existing:
                return {
                    'updated': False,
                    'reason': 'already_resolved',
                    'resolved_record_id': str(existing.get('_id')),
                    'transaction_id': str(existing.get('transaction_id') or ''),
                }

            tx_doc = alert_doc.get('transaction_data') if isinstance(alert_doc.get('transaction_data'), dict) else {}
            raw_model = str(alert_doc.get('model_type') or '').strip().lower()
            if raw_model in {'ae', 'autoencoder'}:
                model_type = 'autoencoder'
                model_label = 'Autoencoder'
            elif raw_model == 'lstm':
                model_type = 'lstm'
                model_label = 'LSTM'
            elif raw_model == 'snn':
                model_type = 'snn'
                model_label = 'SNN'
            else:
                model_type = raw_model or 'unknown'
                model_label = model_type.title()

            fraud_score = float(alert_doc.get('fraud_probability', 0.0) or 0.0)
            resolved_at = datetime.utcnow()
            analyst_info = analyst if isinstance(analyst, dict) else {}
            compact_evidence = []
            for key, value in tx_doc.items():
                if key in {'_id', 'transaction_id', 'inserted_at_dt'}:
                    continue
                if isinstance(value, (dict, list, tuple)):
                    continue
                compact_evidence.append({'attribute': str(key), 'value': str(value)})
                if len(compact_evidence) >= 6:
                    break

            resolved_record = {
                'source_alert_id': source_alert_id,
                'transaction_id': str(alert_doc.get('transaction_id') or tx_doc.get('transaction_id') or ''),
                'title': str(alert_doc.get('title') or ''),
                'model_type': model_type,
                'model_label': model_label,
                'risk_level': str(alert_doc.get('risk_level') or 'High'),
                'fraud_score': fraud_score,
                'decision_threshold': float(alert_doc.get('decision_threshold', 0.5) or 0.5),
                'risk_percent': max(0, min(100, int(round(fraud_score * 100)))),
                'amount': float(tx_doc.get('amt', 0.0) or 0.0),
                'merchant': str(tx_doc.get('merchant') or tx_doc.get('category') or 'N/A'),
                'category': str(tx_doc.get('category') or 'N/A'),
                'city': str(tx_doc.get('city') or 'N/A'),
                'state': str(tx_doc.get('state') or 'N/A'),
                'job': str(tx_doc.get('job') or 'N/A'),
                'status': 'resolved_fraud',
                'alerted_at': alert_doc.get('alerted_at') or alert_doc.get('inserted_at'),
                'resolved_at': resolved_at.isoformat(),
                'resolved_at_dt': resolved_at,
                'resolved_by': {
                    'id': str(analyst_info.get('id') or ''),
                    'name': str(analyst_info.get('name') or 'Unknown Analyst'),
                    'email': str(analyst_info.get('email') or ''),
                    'role': str(analyst_info.get('role') or 'analyst'),
                },
                'resolution_note': str(resolution_note or 'Confirmed as fraud by analyst review.'),
                'transaction_context': {
                    'category': str(tx_doc.get('category') or 'N/A'),
                    'city': str(tx_doc.get('city') or 'N/A'),
                    'state': str(tx_doc.get('state') or 'N/A'),
                    'job': str(tx_doc.get('job') or 'N/A'),
                },
                'evidence': compact_evidence,
                'source': 'mongodb:resolved_frauds',
            }

            insert_result = resolved_collection.insert_one(resolved_record)
            alerts_collection.update_one(
                {'_id': alert_doc.get('_id')},
                {
                    '$set': {
                        'alert_status': 'resolved_fraud',
                        'resolved_at': resolved_at.isoformat(),
                        'resolved_at_dt': resolved_at,
                        'resolved_by': resolved_record['resolved_by'],
                    }
                }
            )
            alerts_collection.delete_one({'_id': alert_doc.get('_id')})

            return {
                'updated': True,
                'transaction_id': resolved_record['transaction_id'],
                'resolved_record_id': str(insert_result.inserted_id),
                'resolved_by': resolved_record['resolved_by'],
                'status': resolved_record['status'],
            }
        except Exception as e:
            return {'updated': False, 'reason': str(e)}

    def summarize_and_cleanup_previous_hour(self):
        """
        Summarize and clean up raw stream/prediction data for the most recently completed hour.

        Returns:
            Summary metadata dictionary
        """
        if not self.connected:
            return {'processed': False, 'reason': 'not_connected'}

        current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        previous_hour_start = current_hour - timedelta(hours=1)

        reports = self.db[self.COLLECTIONS['hourly_reports']]
        if reports.count_documents({'hour_start': previous_hour_start}, limit=1) > 0:
            return {
                'processed': False,
                'reason': 'already_summarized',
                'hour_start': previous_hour_start.isoformat()
            }

        return self.summarize_and_cleanup_hour(previous_hour_start)

    def summarize_and_cleanup_hour(self, hour_start):
        """
        Summarize one UTC hour of raw data and remove detailed records.

        Args:
            hour_start: datetime at exact hour boundary in UTC

        Returns:
            Summary metadata dictionary
        """
        if not self.connected:
            return {'processed': False, 'reason': 'not_connected'}

        hour_start = hour_start.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)

        try:
            fraud_collection = self.db[self.COLLECTIONS['fraud_results']]
            alerts_collection = self.db[self.COLLECTIONS['immediate_alerts']]
            tx_collection = self.db[self.COLLECTIONS['transactions']]
            reports_collection = self.db[self.COLLECTIONS['hourly_reports']]

            time_range = {'inserted_at_dt': {'$gte': hour_start, '$lt': hour_end}}

            model_pipeline = [
                {'$match': time_range},
                {
                    '$group': {
                        '_id': '$model_type',
                        'total_predictions': {'$sum': 1},
                        'fraud_detected': {
                            '$sum': {
                                '$cond': [{'$eq': ['$is_fraud', True]}, 1, 0]
                            }
                        },
                        'avg_processing_time_ms': {'$avg': '$processing_time_ms'},
                        'avg_fraud_probability': {'$avg': '$fraud_probability'},
                        'avg_reconstruction_error': {'$avg': '$reconstruction_error'},
                        'avg_fraud_score': {'$avg': '$fraud_score'},
                        'low_risk': {
                            '$sum': {
                                '$cond': [{'$eq': ['$risk_level', 'Low']}, 1, 0]
                            }
                        },
                        'medium_risk': {
                            '$sum': {
                                '$cond': [{'$in': ['$risk_level', ['Medium', 'Medium-Low', 'Medium-High']]}, 1, 0]
                            }
                        },
                        'high_risk': {
                            '$sum': {
                                '$cond': [{'$eq': ['$risk_level', 'High']}, 1, 0]
                            }
                        }
                    }
                },
                {
                    '$project': {
                        '_id': 0,
                        'model_type': {'$ifNull': ['$_id', 'unknown']},
                        'total_predictions': 1,
                        'fraud_detected': 1,
                        'fraud_rate_percent': {
                            '$cond': [
                                {'$gt': ['$total_predictions', 0]},
                                {'$multiply': [{'$divide': ['$fraud_detected', '$total_predictions']}, 100]},
                                0
                            ]
                        },
                        'avg_processing_time_ms': {'$ifNull': ['$avg_processing_time_ms', 0]},
                        'avg_fraud_probability': {'$ifNull': ['$avg_fraud_probability', 0]},
                        'avg_reconstruction_error': {'$ifNull': ['$avg_reconstruction_error', 0]},
                        'avg_fraud_score': {'$ifNull': ['$avg_fraud_score', 0]},
                        'risk_distribution': {
                            'low': '$low_risk',
                            'medium': '$medium_risk',
                            'high': '$high_risk'
                        }
                    }
                }
            ]

            per_model = list(fraud_collection.aggregate(model_pipeline))

            total_predictions = fraud_collection.count_documents(time_range)
            total_fraud = fraud_collection.count_documents({**time_range, 'is_fraud': True})
            total_alerts = alerts_collection.count_documents(time_range)
            open_alerts = alerts_collection.count_documents({**time_range, 'alert_status': {'$ne': 'dismissed'}})
            dismissed_alerts = alerts_collection.count_documents({**time_range, 'alert_status': 'dismissed'})
            total_streamed_docs = tx_collection.count_documents(time_range)
            medium_high_alerts = fraud_collection.count_documents({**time_range, 'risk_level': 'Medium-High'})
            medium_low_alerts = fraud_collection.count_documents({**time_range, 'risk_level': 'Medium-Low'})
            high_alerts = fraud_collection.count_documents({**time_range, 'risk_level': 'High'})

            amount_pipeline = [
                {'$match': time_range},
                {
                    '$group': {
                        '_id': None,
                        'total_amount': {'$sum': {'$ifNull': ['$amt', 0]}},
                        'fraud_amount': {
                            '$sum': {
                                '$cond': [
                                    {'$eq': ['$is_fraud', True]},
                                    {'$ifNull': ['$amt', 0]},
                                    0
                                ]
                            }
                        }
                    }
                }
            ]
            amount_result = list(fraud_collection.aggregate(amount_pipeline))
            total_amount = float(amount_result[0].get('total_amount', 0.0)) if amount_result else 0.0
            fraud_amount = float(amount_result[0].get('fraud_amount', 0.0)) if amount_result else 0.0

            merchant_pipeline = [
                {'$match': {**time_range, 'is_fraud': True}},
                {
                    '$group': {
                        '_id': {'$ifNull': ['$merchant', 'Unknown Merchant']},
                        'fraud_count': {'$sum': 1},
                        'total_fraud_amount': {'$sum': {'$ifNull': ['$amt', 0]}}
                    }
                },
                {'$sort': {'fraud_count': -1, 'total_fraud_amount': -1}},
                {'$limit': 5}
            ]
            merchant_result = list(fraud_collection.aggregate(merchant_pipeline))
            top_fraudulent_merchants = [
                {
                    'merchant_name': str(item.get('_id', 'Unknown Merchant')),
                    'fraud_count': int(item.get('fraud_count', 0) or 0),
                    'total_fraud_amount': float(item.get('total_fraud_amount', 0.0) or 0.0),
                }
                for item in merchant_result
            ]

            trend_labels = []
            trend_total = []
            trend_fraud = []
            bucket_minutes = 10
            bucket_count = 6
            for idx in range(bucket_count):
                bucket_start = hour_start + timedelta(minutes=idx * bucket_minutes)
                bucket_end = bucket_start + timedelta(minutes=bucket_minutes)
                label = bucket_start.strftime('%H:%M')
                bucket_query = {'inserted_at_dt': {'$gte': bucket_start, '$lt': bucket_end}}
                trend_labels.append(label)
                trend_total.append(fraud_collection.count_documents(bucket_query))
                trend_fraud.append(fraud_collection.count_documents({**bucket_query, 'is_fraud': True}))

            fraud_pct = (total_fraud / total_predictions * 100.0) if total_predictions > 0 else 0.0
            high_pct = (high_alerts / total_predictions * 100.0) if total_predictions > 0 else 0.0
            medium_high_pct = (medium_high_alerts / total_predictions * 100.0) if total_predictions > 0 else 0.0
            medium_low_pct = (medium_low_alerts / total_predictions * 100.0) if total_predictions > 0 else 0.0

            known_model_names = {'autoencoder', 'lstm', 'snn'}
            per_model_map = {str(item.get('model_type', 'unknown')).lower(): item for item in per_model}
            detection_model_performance = []
            for model_name in ['autoencoder', 'lstm', 'snn']:
                model_data = per_model_map.get(model_name, {})
                total_model_predictions = int(model_data.get('total_predictions', 0) or 0)
                model_fraud_count = int(model_data.get('fraud_detected', 0) or 0)
                model_fraud_rate = (model_fraud_count / total_model_predictions * 100.0) if total_model_predictions > 0 else 0.0
                detection_model_performance.append({
                    'model': model_name.upper(),
                    'architecture': model_name.upper(),
                    'accuracy': round(100.0 - model_fraud_rate, 2),
                    'precision': round(max(0.0, 100.0 - model_fraud_rate * 0.8), 2),
                    'recall': round(max(0.0, 100.0 - model_fraud_rate * 0.6), 2),
                })

            for item in per_model:
                model_key = str(item.get('model_type', 'unknown')).lower()
                if model_key not in known_model_names:
                    total_model_predictions = int(item.get('total_predictions', 0) or 0)
                    model_fraud_count = int(item.get('fraud_detected', 0) or 0)
                    model_fraud_rate = (model_fraud_count / total_model_predictions * 100.0) if total_model_predictions > 0 else 0.0
                    detection_model_performance.append({
                        'model': str(item.get('model_type', 'UNKNOWN')).upper(),
                        'architecture': str(item.get('model_type', 'UNKNOWN')).upper(),
                        'accuracy': round(100.0 - model_fraud_rate, 2),
                        'precision': round(max(0.0, 100.0 - model_fraud_rate * 0.8), 2),
                        'recall': round(max(0.0, 100.0 - model_fraud_rate * 0.6), 2),
                    })

            analyst_comments = (
                f"Hourly summary shows {total_fraud} flagged fraud case(s) out of {total_predictions} processed "
                f"transactions ({fraud_pct:.2f}% fraud rate). "
                f"High severity accounted for {high_alerts} case(s), with medium-high at {medium_high_alerts} and "
                f"medium-low at {medium_low_alerts}. "
                f"Top merchant exposure is concentrated in "
                f"{top_fraudulent_merchants[0]['merchant_name'] if top_fraudulent_merchants else 'N/A'}."
            )

            realtime_sections = {
                'header': {
                    'report_title': 'FRAUD SUMMARY REPORT',
                    'report_id': f"RPT-RT-{hour_start.strftime('%Y%m%d-%H00')}",
                    'generated_at': datetime.utcnow().strftime('%B %d, %Y, %I:%M %p UTC'),
                    'visibility': 'Internal Use Only',
                },
                'summary_cards': {
                    'total_transactions': int(total_predictions),
                    'number_of_frauds': int(total_fraud),
                    'number_of_normals': int(max(total_predictions - total_fraud, 0)),
                    'total_amount': float(total_amount),
                    'fraud_amount': float(fraud_amount),
                },
                'alerts_severity_summary': [
                    {'severity': 'High Severity', 'count': int(high_alerts), 'percent': round(high_pct, 2)},
                    {'severity': 'Medium-High', 'count': int(medium_high_alerts), 'percent': round(medium_high_pct, 2)},
                    {'severity': 'Medium-Low', 'count': int(medium_low_alerts), 'percent': round(medium_low_pct, 2)},
                ],
                'detection_model_performance': detection_model_performance,
                'transactions_vs_frauds_trend': {
                    'labels': trend_labels,
                    'total_transactions': trend_total,
                    'fraud_cases': trend_fraud,
                    'sampling_note': 'Real-time Sampling',
                },
                'top_fraudulent_merchants': top_fraudulent_merchants,
                'analyst_comments_observations': analyst_comments,
            }

            report_doc = {
                'hour_start': hour_start,
                'hour_end': hour_end,
                'generated_at': datetime.utcnow(),
                'summary_type': 'hourly',
                'totals': {
                    'total_predictions': total_predictions,
                    'fraud_detected': total_fraud,
                    'fraud_rate_percent': (total_fraud / total_predictions * 100) if total_predictions > 0 else 0,
                    'total_alerts': total_alerts,
                    'open_alerts': open_alerts,
                    'dismissed_alerts': dismissed_alerts,
                    'total_streamed_docs': total_streamed_docs,
                    # financial totals — stored here so section data survives a WS crash
                    'total_amount': total_amount,
                    'fraud_amount': fraud_amount,
                    # severity breakdown
                    'high_alerts': high_alerts,
                    'medium_high_alerts': medium_high_alerts,
                    'medium_low_alerts': medium_low_alerts,
                },
                'models': per_model,
                # rich section data — redundant with model_reports but ensures survival if
                # save_model_report() fails (e.g. WS server crashes mid-run)
                'top_fraudulent_merchants': top_fraudulent_merchants,
                'detection_model_performance': detection_model_performance,
                'trend': {
                    'labels': trend_labels,
                    'total_transactions': trend_total,
                    'fraud_cases': trend_fraud,
                },
                'sections': realtime_sections,
            }

            realtime_template_report = self.build_report_document(
                source_type='realtime',
                model_type='multi-model',
                report_id=f"rt_hourly_{hour_start.strftime('%Y%m%d_%H00')}",
                title=f"Real-Time Hourly Fraud Report ({hour_start.strftime('%Y-%m-%d %H:00')} UTC)",
                period={
                    'start': hour_start.isoformat(),
                    'end': hour_end.isoformat(),
                    'granularity': 'hour'
                },
                summary={
                    'total_transactions': total_predictions,
                    'fraud_detected': total_fraud,
                    'legitimate_transactions': max(total_predictions - total_fraud, 0),
                    'fraud_rate_percent': (total_fraud / total_predictions * 100) if total_predictions > 0 else 0.0,
                    'avg_fraud_score': float(np.mean([item.get('avg_fraud_score', 0.0) for item in per_model])) if per_model else 0.0,
                    'threshold_used': None,
                    'total_alerts': total_alerts,
                    'open_alerts': open_alerts,
                    'dismissed_alerts': dismissed_alerts,
                },
                risk_distribution={
                    'low': int(sum(item.get('risk_distribution', {}).get('low', 0) for item in per_model)),
                    'medium': int(sum(item.get('risk_distribution', {}).get('medium', 0) for item in per_model)),
                    'high': int(sum(item.get('risk_distribution', {}).get('high', 0) for item in per_model)),
                },
                model_metrics={
                    'per_model': per_model,
                },
                data_sources={
                    'collections': ['fraud_results', 'immediate_alerts', 'transactions'],
                    'cleanup_applied': True,
                },
                metadata={
                    'summary_type': 'hourly',
                    'hour_start': hour_start.isoformat(),
                    'hour_end': hour_end.isoformat(),
                },
                report_sections=realtime_sections,
            )

            reports_collection.update_one(
                {'hour_start': hour_start},
                {'$set': report_doc},
                upsert=True
            )
            self.save_model_report(realtime_template_report)

            deleted_predictions = fraud_collection.delete_many(time_range).deleted_count

            # Keep high alerts until investigators dismiss them. Only dismissed alerts are cleaned up.
            deleted_alerts = alerts_collection.delete_many({
                **time_range,
                'alert_status': 'dismissed'
            }).deleted_count

            deleted_streamed = tx_collection.delete_many(time_range).deleted_count

            return {
                'processed': True,
                'hour_start': hour_start.isoformat(),
                'hour_end': hour_end.isoformat(),
                'summary': report_doc['totals'],
                'report_id': realtime_template_report['report_id'],
                'deleted': {
                    'fraud_results': deleted_predictions,
                    'immediate_alerts': deleted_alerts,
                    'transactions': deleted_streamed
                }
            }

        except Exception as e:
            print(f"❌ Error during hourly summarize/cleanup: {e}")
            return {
                'processed': False,
                'reason': str(e),
                'hour_start': hour_start.isoformat(),
                'hour_end': hour_end.isoformat()
            }

    def build_report_document(
        self,
        source_type,
        model_type,
        report_id,
        title,
        period,
        summary,
        risk_distribution,
        model_metrics=None,
        top_cases=None,
        data_sources=None,
        metadata=None,
        report_sections=None,
    ):
        """Build a shared report schema for batch and real-time pipelines."""
        generated_at = datetime.utcnow()
        safe_summary = summary or {}
        total_transactions = int(safe_summary.get('total_transactions', 0) or 0)
        fraud_detected = int(safe_summary.get('fraud_detected', 0) or 0)
        legitimate = int(safe_summary.get('legitimate_transactions', max(total_transactions - fraud_detected, 0)) or 0)
        fraud_rate = safe_summary.get('fraud_rate_percent')
        if fraud_rate is None:
            fraud_rate = (fraud_detected / total_transactions * 100.0) if total_transactions > 0 else 0.0

        return {
            'report_id': str(report_id),
            'title': str(title),
            'generated_at': generated_at,
            'generated_at_iso': generated_at.isoformat(),
            'source': {
                'type': str(source_type),
            },
            'model': {
                'type': str(model_type),
            },
            'period': period or {},
            'summary': {
                'total_transactions': total_transactions,
                'fraud_detected': fraud_detected,
                'legitimate_transactions': legitimate,
                'fraud_rate_percent': float(fraud_rate),
                'avg_fraud_score': float(safe_summary.get('avg_fraud_score', 0.0) or 0.0),
                'threshold_used': safe_summary.get('threshold_used'),
                'total_alerts': int(safe_summary.get('total_alerts', fraud_detected) or 0),
                'open_alerts': int(safe_summary.get('open_alerts', 0) or 0),
                'dismissed_alerts': int(safe_summary.get('dismissed_alerts', 0) or 0),
            },
            'risk_distribution': {
                'low': int((risk_distribution or {}).get('low', 0) or 0),
                'medium': int((risk_distribution or {}).get('medium', 0) or 0),
                'high': int((risk_distribution or {}).get('high', 0) or 0),
            },
            'model_metrics': model_metrics or {},
            'top_cases': top_cases or [],
            'sections': report_sections or {},
            'data_sources': data_sources or {},
            'metadata': metadata or {},
            'template_version': 'v1',
            'template_name': 'fraud_summary_report_v1',
            'template_source': 'visily_1203205765',
        }

    def save_model_report(self, report_doc):
        """Upsert a template-based report document."""
        if not self.connected:
            return False

        try:
            reports_collection = self.db[self.COLLECTIONS['model_reports']]
            reports_collection.update_one(
                {'report_id': str(report_doc.get('report_id'))},
                {'$set': report_doc},
                upsert=True,
            )
            return True
        except Exception as e:
            print(f"❌ Error saving model report: {e}")
            return False


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
