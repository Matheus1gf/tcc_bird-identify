#!/usr/bin/env python3
"""
Sistema de Otimização de Performance
Implementa lazy loading e cache para melhorar performance
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Otimizador de performance com lazy loading e cache"""
    
    def __init__(self):
        self.cache = {}
        self.lazy_components = {}
        self.initialization_times = {}
    
    def lazy_load(self, component_name: str, loader_func: Callable) -> Any:
        """
        Carrega componente apenas quando necessário
        """
        if component_name not in self.lazy_components:
            start_time = time.time()
            try:
                self.lazy_components[component_name] = loader_func()
                load_time = time.time() - start_time
                self.initialization_times[component_name] = load_time
                logger.info(f"Componente {component_name} carregado em {load_time:.2f}s")
            except Exception as e:
                logger.error(f"Erro ao carregar {component_name}: {e}")
                self.lazy_components[component_name] = None
        
        return self.lazy_components[component_name]
    
    def get_component(self, component_name: str) -> Optional[Any]:
        """Retorna componente carregado"""
        return self.lazy_components.get(component_name)
    
    def is_loaded(self, component_name: str) -> bool:
        """Verifica se componente foi carregado"""
        return component_name in self.lazy_components
    
    def get_initialization_time(self, component_name: str) -> float:
        """Retorna tempo de inicialização do componente"""
        return self.initialization_times.get(component_name, 0.0)
    
    def get_total_initialization_time(self) -> float:
        """Retorna tempo total de inicialização"""
        return sum(self.initialization_times.values())
    
    def clear_cache(self):
        """Limpa cache"""
        self.cache.clear()
        self.lazy_components.clear()
        self.initialization_times.clear()

# Instância global do otimizador
performance_optimizer = PerformanceOptimizer()

def lazy_loading(component_name: str):
    """
    Decorator para lazy loading de componentes
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return performance_optimizer.lazy_load(component_name, lambda: func(*args, **kwargs))
        return wrapper
    return decorator

def cached_result(cache_key: str, ttl: int = 3600):
    """
    Decorator para cache de resultados
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Criar chave única baseada nos argumentos
            key = f"{cache_key}_{hash(str(args) + str(kwargs))}"
            
            # Verificar cache
            if key in performance_optimizer.cache:
                cached_data, timestamp = performance_optimizer.cache[key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit para {cache_key}")
                    return cached_data
            
            # Executar função e cachear resultado
            result = func(*args, **kwargs)
            performance_optimizer.cache[key] = (result, time.time())
            logger.debug(f"Resultado cacheado para {cache_key}")
            
            return result
        return wrapper
    return decorator

def measure_time(func_name: str):
    """
    Decorator para medir tempo de execução
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(f"{func_name} executado em {execution_time:.2f}s")
            return result
        return wrapper
    return decorator

class ComponentManager:
    """Gerenciador de componentes com lazy loading"""
    
    def __init__(self):
        self.components = {}
        self.loaders = {}
    
    def register_component(self, name: str, loader_func: Callable):
        """Registra um componente para lazy loading"""
        self.loaders[name] = loader_func
        logger.debug(f"Componente {name} registrado para lazy loading")
    
    def get_component(self, name: str) -> Optional[Any]:
        """Obtém componente (carrega se necessário)"""
        if name not in self.components:
            if name in self.loaders:
                start_time = time.time()
                try:
                    self.components[name] = self.loaders[name]()
                    load_time = time.time() - start_time
                    logger.info(f"Componente {name} carregado em {load_time:.2f}s")
                except Exception as e:
                    logger.error(f"Erro ao carregar componente {name}: {e}")
                    self.components[name] = None
            else:
                logger.warning(f"Componente {name} não registrado")
                return None
        
        return self.components[name]
    
    def is_component_loaded(self, name: str) -> bool:
        """Verifica se componente está carregado"""
        return name in self.components
    
    def preload_component(self, name: str):
        """Pré-carrega um componente"""
        if not self.is_component_loaded(name):
            self.get_component(name)
    
    def preload_critical_components(self, component_names: list):
        """Pré-carrega componentes críticos"""
        logger.info("Pré-carregando componentes críticos...")
        start_time = time.time()
        
        for name in component_names:
            self.preload_component(name)
        
        total_time = time.time() - start_time
        logger.info(f"Componentes críticos pré-carregados em {total_time:.2f}s")

# Instância global do gerenciador
component_manager = ComponentManager()

def optimize_imports():
    """
    Otimiza imports para melhorar tempo de inicialização
    """
    logger.info("Otimizando imports...")
    
    # Imports críticos (sempre necessários)
    critical_imports = [
        'os', 'cv2', 'numpy', 'PIL', 'typing', 'logging', 'json', 'time'
    ]
    
    # Imports opcionais (lazy loading)
    optional_imports = [
        'tensorflow', 'keras', 'ultralytics', 'sklearn', 'matplotlib'
    ]
    
    logger.info(f"Imports críticos: {len(critical_imports)}")
    logger.info(f"Imports opcionais: {len(optional_imports)}")

def get_performance_stats() -> Dict[str, Any]:
    """
    Retorna estatísticas de performance
    """
    return {
        'total_initialization_time': performance_optimizer.get_total_initialization_time(),
        'component_times': performance_optimizer.initialization_times,
        'loaded_components': list(performance_optimizer.lazy_components.keys()),
        'cache_size': len(performance_optimizer.cache)
    }

def log_performance_summary():
    """
    Loga resumo de performance
    """
    stats = get_performance_stats()
    
    logger.info("=== RESUMO DE PERFORMANCE ===")
    logger.info(f"Tempo total de inicialização: {stats['total_initialization_time']:.2f}s")
    logger.info(f"Componentes carregados: {len(stats['loaded_components'])}")
    logger.info(f"Tamanho do cache: {stats['cache_size']}")
    
    if stats['component_times']:
        logger.info("Tempos por componente:")
        for component, time_taken in stats['component_times'].items():
            logger.info(f"  {component}: {time_taken:.2f}s")

# Função de conveniência para otimização rápida
def quick_optimize():
    """
    Aplica otimizações rápidas
    """
    logger.info("Aplicando otimizações rápidas...")
    
    # Otimizar imports
    optimize_imports()
    
    # Configurar logging para performance
    logging.getLogger().setLevel(logging.WARNING)
    
    logger.info("Otimizações rápidas aplicadas")
