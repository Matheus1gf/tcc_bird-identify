#!/usr/bin/env python3
"""
Teste de Performance - Sistema de Identificação de Pássaros
Testa performance, carga e escalabilidade do sistema
"""

import requests
import time
import json
import os
import sys
from datetime import datetime
import threading
import statistics
import psutil
import subprocess

class PerformanceTester:
    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url
        self.results = []
        self.performance_metrics = {}
        
    def log_test(self, test_name, status, details=""):
        """Registrar resultado do teste"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {status}")
        if details:
            print(f"   📝 {details}")
    
    def test_response_time(self):
        """Teste 1: Tempo de resposta"""
        try:
            response_times = []
            
            # Fazer múltiplas requisições para calcular tempo médio
            for i in range(10):
                start_time = time.time()
                response = requests.get(self.base_url, timeout=30)
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append(end_time - start_time)
                time.sleep(0.1)
            
            if response_times:
                avg_time = statistics.mean(response_times)
                min_time = min(response_times)
                max_time = max(response_times)
                
                self.performance_metrics['response_time'] = {
                    'average': avg_time,
                    'min': min_time,
                    'max': max_time,
                    'samples': len(response_times)
                }
                
                if avg_time < 5.0:
                    self.log_test("Tempo de Resposta", "PASS", f"Média: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
                    return True
                else:
                    self.log_test("Tempo de Resposta", "FAIL", f"Média: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
                    return False
            else:
                self.log_test("Tempo de Resposta", "FAIL", "Nenhuma resposta válida recebida")
                return False
        except Exception as e:
            self.log_test("Tempo de Resposta", "FAIL", str(e))
            return False
    
    def test_throughput(self):
        """Teste 2: Taxa de transferência"""
        try:
            # Testar throughput com múltiplas requisições simultâneas
            start_time = time.time()
            success_count = 0
            total_requests = 20
            
            def make_request():
                nonlocal success_count
                try:
                    response = requests.get(self.base_url, timeout=10)
                    if response.status_code == 200:
                        success_count += 1
                except:
                    pass
            
            # Fazer requisições simultâneas
            threads = []
            for i in range(total_requests):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()
            
            # Aguardar todas as threads
            for thread in threads:
                thread.join()
            
            end_time = time.time()
            total_time = end_time - start_time
            throughput = success_count / total_time
            
            self.performance_metrics['throughput'] = {
                'requests_per_second': throughput,
                'total_requests': total_requests,
                'successful_requests': success_count,
                'total_time': total_time
            }
            
            if throughput >= 2.0:
                self.log_test("Taxa de Transferência", "PASS", f"{throughput:.2f} req/s, {success_count}/{total_requests} sucessos")
                return True
            else:
                self.log_test("Taxa de Transferência", "FAIL", f"{throughput:.2f} req/s, {success_count}/{total_requests} sucessos")
                return False
        except Exception as e:
            self.log_test("Taxa de Transferência", "FAIL", str(e))
            return False
    
    def test_concurrent_users(self):
        """Teste 3: Usuários concorrentes"""
        try:
            # Simular usuários concorrentes
            concurrent_users = 5
            success_count = 0
            total_requests = concurrent_users * 4  # 4 requisições por usuário
            
            def simulate_user(user_id):
                nonlocal success_count
                for i in range(4):
                    try:
                        response = requests.get(self.base_url, timeout=10)
                        if response.status_code == 200:
                            success_count += 1
                        time.sleep(0.5)
                    except:
                        pass
            
            # Simular usuários concorrentes
            threads = []
            for user_id in range(concurrent_users):
                thread = threading.Thread(target=simulate_user, args=(user_id,))
                threads.append(thread)
                thread.start()
            
            # Aguardar todas as threads
            for thread in threads:
                thread.join()
            
            success_rate = success_count / total_requests
            
            self.performance_metrics['concurrent_users'] = {
                'concurrent_users': concurrent_users,
                'total_requests': total_requests,
                'successful_requests': success_count,
                'success_rate': success_rate
            }
            
            if success_rate >= 0.8:
                self.log_test("Usuários Concorrentes", "PASS", f"{concurrent_users} usuários, {success_rate:.2%} taxa de sucesso")
                return True
            else:
                self.log_test("Usuários Concorrentes", "FAIL", f"{concurrent_users} usuários, {success_rate:.2%} taxa de sucesso")
                return False
        except Exception as e:
            self.log_test("Usuários Concorrentes", "FAIL", str(e))
            return False
    
    def test_memory_usage(self):
        """Teste 4: Uso de memória"""
        try:
            # Verificar uso de memória do sistema
            memory_info = psutil.virtual_memory()
            memory_usage_percent = memory_info.percent
            available_memory = memory_info.available / (1024**3)  # GB
            
            self.performance_metrics['memory_usage'] = {
                'usage_percent': memory_usage_percent,
                'available_gb': available_memory,
                'total_gb': memory_info.total / (1024**3)
            }
            
            if memory_usage_percent < 90:
                self.log_test("Uso de Memória", "PASS", f"{memory_usage_percent:.1f}% usado, {available_memory:.1f}GB disponível")
                return True
            else:
                self.log_test("Uso de Memória", "WARN", f"{memory_usage_percent:.1f}% usado, {available_memory:.1f}GB disponível")
                return False
        except Exception as e:
            self.log_test("Uso de Memória", "FAIL", str(e))
            return False
    
    def test_cpu_usage(self):
        """Teste 5: Uso de CPU"""
        try:
            # Verificar uso de CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            self.performance_metrics['cpu_usage'] = {
                'usage_percent': cpu_percent
            }
            
            if cpu_percent < 80:
                self.log_test("Uso de CPU", "PASS", f"{cpu_percent:.1f}% usado")
                return True
            else:
                self.log_test("Uso de CPU", "WARN", f"{cpu_percent:.1f}% usado")
                return False
        except Exception as e:
            self.log_test("Uso de CPU", "FAIL", str(e))
            return False
    
    def test_load_time(self):
        """Teste 6: Tempo de carregamento"""
        try:
            load_times = []
            
            # Testar tempo de carregamento com diferentes tamanhos de conteúdo
            for i in range(5):
                start_time = time.time()
                response = requests.get(self.base_url, timeout=30)
                end_time = time.time()
                
                if response.status_code == 200:
                    load_time = end_time - start_time
                    content_size = len(response.content)
                    load_times.append({
                        'time': load_time,
                        'size': content_size
                    })
                time.sleep(1)
            
            if load_times:
                avg_load_time = statistics.mean([lt['time'] for lt in load_times])
                avg_content_size = statistics.mean([lt['size'] for lt in load_times])
                
                self.performance_metrics['load_time'] = {
                    'average_time': avg_load_time,
                    'average_size': avg_content_size,
                    'samples': len(load_times)
                }
                
                if avg_load_time < 3.0:
                    self.log_test("Tempo de Carregamento", "PASS", f"Média: {avg_load_time:.3f}s, Tamanho: {avg_content_size:.0f} bytes")
                    return True
                else:
                    self.log_test("Tempo de Carregamento", "FAIL", f"Média: {avg_load_time:.3f}s, Tamanho: {avg_content_size:.0f} bytes")
                    return False
            else:
                self.log_test("Tempo de Carregamento", "FAIL", "Nenhuma medição válida")
                return False
        except Exception as e:
            self.log_test("Tempo de Carregamento", "FAIL", str(e))
            return False
    
    def test_scalability(self):
        """Teste 7: Escalabilidade"""
        try:
            # Testar escalabilidade com diferentes cargas
            scalability_results = []
            
            for load_level in [1, 5, 10]:
                start_time = time.time()
                success_count = 0
                total_requests = load_level * 2
                
                def make_request():
                    nonlocal success_count
                    try:
                        response = requests.get(self.base_url, timeout=5)
                        if response.status_code == 200:
                            success_count += 1
                    except:
                        pass
                
                # Fazer requisições simultâneas
                threads = []
                for i in range(total_requests):
                    thread = threading.Thread(target=make_request)
                    threads.append(thread)
                    thread.start()
                
                # Aguardar todas as threads
                for thread in threads:
                    thread.join()
                
                end_time = time.time()
                total_time = end_time - start_time
                success_rate = success_count / total_requests
                
                scalability_results.append({
                    'load_level': load_level,
                    'success_rate': success_rate,
                    'total_time': total_time
                })
                
                time.sleep(1)
            
            # Verificar se a escalabilidade é mantida
            success_rates = [sr['success_rate'] for sr in scalability_results]
            min_success_rate = min(success_rates)
            
            self.performance_metrics['scalability'] = {
                'results': scalability_results,
                'min_success_rate': min_success_rate
            }
            
            if min_success_rate >= 0.7:
                self.log_test("Escalabilidade", "PASS", f"Taxa mínima de sucesso: {min_success_rate:.2%}")
                return True
            else:
                self.log_test("Escalabilidade", "FAIL", f"Taxa mínima de sucesso: {min_success_rate:.2%}")
                return False
        except Exception as e:
            self.log_test("Escalabilidade", "FAIL", str(e))
            return False
    
    def test_stress_test(self):
        """Teste 8: Teste de estresse"""
        try:
            # Teste de estresse com alta carga
            stress_duration = 30  # segundos
            start_time = time.time()
            success_count = 0
            total_requests = 0
            
            while time.time() - start_time < stress_duration:
                try:
                    response = requests.get(self.base_url, timeout=5)
                    total_requests += 1
                    if response.status_code == 200:
                        success_count += 1
                except:
                    total_requests += 1
                
                time.sleep(0.1)
            
            success_rate = success_count / total_requests if total_requests > 0 else 0
            
            self.performance_metrics['stress_test'] = {
                'duration': stress_duration,
                'total_requests': total_requests,
                'successful_requests': success_count,
                'success_rate': success_rate
            }
            
            if success_rate >= 0.8:
                self.log_test("Teste de Estresse", "PASS", f"{stress_duration}s, {success_rate:.2%} taxa de sucesso")
                return True
            else:
                self.log_test("Teste de Estresse", "FAIL", f"{stress_duration}s, {success_rate:.2%} taxa de sucesso")
                return False
        except Exception as e:
            self.log_test("Teste de Estresse", "FAIL", str(e))
            return False
    
    def test_resource_efficiency(self):
        """Teste 9: Eficiência de recursos"""
        try:
            # Verificar eficiência de recursos
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Fazer uma requisição e medir recursos
            start_time = time.time()
            response = requests.get(self.base_url, timeout=10)
            end_time = time.time()
            
            response_time = end_time - start_time
            content_size = len(response.content)
            
            # Calcular eficiência
            efficiency_score = content_size / response_time if response_time > 0 else 0
            
            self.performance_metrics['resource_efficiency'] = {
                'response_time': response_time,
                'content_size': content_size,
                'efficiency_score': efficiency_score,
                'memory_usage': memory_info.percent,
                'cpu_usage': cpu_percent
            }
            
            if efficiency_score > 1000:  # bytes por segundo
                self.log_test("Eficiência de Recursos", "PASS", f"Score: {efficiency_score:.0f} bytes/s")
                return True
            else:
                self.log_test("Eficiência de Recursos", "FAIL", f"Score: {efficiency_score:.0f} bytes/s")
                return False
        except Exception as e:
            self.log_test("Eficiência de Recursos", "FAIL", str(e))
            return False
    
    def test_performance_consistency(self):
        """Teste 10: Consistência de performance"""
        try:
            # Testar consistência de performance ao longo do tempo
            performance_samples = []
            
            for i in range(10):
                start_time = time.time()
                response = requests.get(self.base_url, timeout=10)
                end_time = time.time()
                
                if response.status_code == 200:
                    response_time = end_time - start_time
                    performance_samples.append(response_time)
                
                time.sleep(1)
            
            if performance_samples:
                avg_performance = statistics.mean(performance_samples)
                std_performance = statistics.stdev(performance_samples)
                coefficient_of_variation = std_performance / avg_performance if avg_performance > 0 else 0
                
                self.performance_metrics['performance_consistency'] = {
                    'average': avg_performance,
                    'standard_deviation': std_performance,
                    'coefficient_of_variation': coefficient_of_variation,
                    'samples': len(performance_samples)
                }
                
                if coefficient_of_variation < 0.5:  # Menos de 50% de variação
                    self.log_test("Consistência de Performance", "PASS", f"CV: {coefficient_of_variation:.3f}")
                    return True
                else:
                    self.log_test("Consistência de Performance", "FAIL", f"CV: {coefficient_of_variation:.3f}")
                    return False
            else:
                self.log_test("Consistência de Performance", "FAIL", "Nenhuma amostra válida")
                return False
        except Exception as e:
            self.log_test("Consistência de Performance", "FAIL", str(e))
            return False
    
    def run_all_tests(self):
        """Executar todos os testes de performance"""
        print("⚡ INICIANDO TESTES DE PERFORMANCE")
        print("=" * 60)
        
        tests = [
            self.test_response_time,
            self.test_throughput,
            self.test_concurrent_users,
            self.test_memory_usage,
            self.test_cpu_usage,
            self.test_load_time,
            self.test_scalability,
            self.test_stress_test,
            self.test_resource_efficiency,
            self.test_performance_consistency
        ]
        
        passed = 0
        failed = 0
        warned = 0
        
        for test in tests:
            try:
                result = test()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                self.log_test(test.__name__, "FAIL", str(e))
                failed += 1
            
            time.sleep(1)  # Pausa entre testes
        
        # Resumo
        print("\n" + "=" * 60)
        print("📊 RESUMO DOS TESTES DE PERFORMANCE")
        print("=" * 60)
        print(f"✅ Passou: {passed}")
        print(f"❌ Falhou: {failed}")
        print(f"⚠️ Avisos: {warned}")
        print(f"📈 Taxa de Sucesso: {(passed/(passed+failed)*100):.1f}%")
        
        # Salvar resultados
        self.save_results()
        
        return passed, failed, warned
    
    def save_results(self):
        """Salvar resultados em arquivo JSON"""
        try:
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "results": self.results,
                "performance_metrics": self.performance_metrics
            }
            
            with open("test_results_performance.json", "w") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados salvos em: test_results_performance.json")
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    print("⚡ TESTE DE PERFORMANCE - SISTEMA DE IDENTIFICAÇÃO DE PÁSSAROS")
    print("=" * 70)
    
    tester = PerformanceTester()
    
    try:
        passed, failed, warned = tester.run_all_tests()
        
        if failed == 0:
            print("\n🎉 TODOS OS TESTES DE PERFORMANCE PASSARAM!")
            sys.exit(0)
        else:
            print(f"\n⚠️ {failed} TESTES DE PERFORMANCE FALHARAM")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Testes interrompidos pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
