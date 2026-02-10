"""
Database module for NeuroDetect
"""
from .mongodb import MongoDB, get_mongodb_instance

__all__ = ['MongoDB', 'get_mongodb_instance']
