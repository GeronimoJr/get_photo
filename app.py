import streamlit as st
import requests
import tempfile
import os
import re
import json
import time
import random
from datetime import datetime
import pandas as pd
import io
import ftplib
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import uuid
import codecs
from bs4 import BeautifulSoup
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import concurrent.futures
import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

# Stałe konfiguracyjne
FTP_CONFIG = {
    'MIN_CONNECTIONS': 1,
    'MAX_CONNECTIONS': 5,
    'MIN_BATCH_SIZE': 2,
    'MAX_BATCH_SIZE': 10,
    'DEFAULT_RETRY_DELAY': 1,
    'MAX_RETRY_DELAY': 5,
    'CONNECTION_TIMEOUT': 60,
    'MIN_WORKERS': 1,
    'MAX_WORKERS': 10,
    'DEFAULT_WORKERS': 3,
    'KEEPALIVE_INTERVAL': 30,  # Interwał w sekundach dla keepalive
    'IDLE_TIMEOUT': 300  # Timeout bezczynności w sekundach (5 minut)
}

class FTPManager:
    _instances = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, settings):
        key = f"{settings['host']}:{settings['username']}:{settings['directory']}"
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(settings)
            return cls._instances[key]
    
    def __init__(self, settings):
        self.settings = settings
        self.ftp = None
        self.connected = False
        self.lock = threading.Lock()
        self.last_activity = time.time()
        self.connection_limit = FTP_CONFIG['MAX_CONNECTIONS']
        self.connection_semaphore = threading.BoundedSemaphore(self.connection_limit)
        self.error_count = 0
        self.last_error = None
        self.keepalive_timer = None
        self.idle_timer = None
        
    def _start_keepalive(self):
        """Rozpoczyna timer keepalive"""
        if self.keepalive_timer:
            self.keepalive_timer.cancel()
        self.keepalive_timer = threading.Timer(FTP_CONFIG['KEEPALIVE_INTERVAL'], self._send_keepalive)
        self.keepalive_timer.daemon = True
        self.keepalive_timer.start()
        
    def _send_keepalive(self):
        """Wysyła komendę NOOP do serwera FTP aby utrzymać połączenie"""
        try:
            with self.lock:
                if self.connected and self.ftp:
                    self.ftp.voidcmd('NOOP')
                    self.last_activity = time.time()
                    self._start_keepalive()
        except Exception as e:
            print(f"Błąd keepalive: {str(e)}")
            self.connected = False
            
    def _start_idle_timer(self):
        """Rozpoczyna timer bezczynności"""
        if self.idle_timer:
            self.idle_timer.cancel()
        self.idle_timer = threading.Timer(FTP_CONFIG['IDLE_TIMEOUT'], self._handle_idle_timeout)
        self.idle_timer.daemon = True
        self.idle_timer.start()
        
    def _handle_idle_timeout(self):
        """Obsługuje timeout bezczynności"""
        with self.lock:
            if time.time() - self.last_activity >= FTP_CONFIG['IDLE_TIMEOUT']:
                self.close()
                
    def verify_connection(self):
        """Weryfikuje połączenie FTP i zwraca szczegółowy status"""
        try:
            if not self.connect():
                return False, "Nie można nawiązać połączenia FTP"
                
            # Testuj uprawnienia do zapisu
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(b"test")
                tmp.seek(0)
                try:
                    test_filename = f"test_{uuid.uuid4().hex[:8]}.tmp"
                    self.ftp.storbinary(f'STOR {test_filename}', tmp)
                    self.ftp.delete(test_filename)
                except Exception as e:
                    return False, f"Brak uprawnień do zapisu: {str(e)}"
                
            return True, "Połączenie FTP działa prawidłowo"
        except Exception as e:
            return False, f"Błąd weryfikacji FTP: {str(e)}"
    
    def connect(self):
        with self.lock:
            if self.connected and time.time() - self.last_activity < FTP_CONFIG['CONNECTION_TIMEOUT']:
                return True
                
            try:
                if self.ftp:
                    try:
                        self.ftp.quit()
                    except Exception as e:
                        print(f"Błąd podczas zamykania poprzedniego połączenia: {str(e)}")
                
                # Sprawdź czy używamy FTPS
                use_tls = self.settings.get("use_tls", False)
                if use_tls:
                    self.ftp = ftplib.FTP_TLS()
                else:
                    self.ftp = ftplib.FTP()
                
                # Ustaw timeout połączenia
                self.ftp.connect(
                    self.settings["host"], 
                    self.settings["port"], 
                    timeout=FTP_CONFIG['CONNECTION_TIMEOUT']
                )
                
                # Logowanie i konfiguracja TLS
                if use_tls:
                    self.ftp.auth()
                    self.ftp.prot_p()  # Włącz ochronę danych
                
                self.ftp.login(self.settings["username"], self.settings["password"])
                
                # Optymalizacja ustawień FTP
                self.ftp.set_pasv(True)  # Tryb pasywny często jest bardziej niezawodny
                
                if self.settings["directory"] and self.settings["directory"] != "/":
                    try:
                        self.ftp.cwd(self.settings["directory"])
                    except ftplib.error_perm as e:
                        dirs = [d for d in self.settings["directory"].strip("/").split("/") if d]
                        for directory in dirs:
                            try:
                                self.ftp.cwd(directory)
                            except ftplib.error_perm:
                                self.ftp.mkd(directory)
                                self.ftp.cwd(directory)
                
                self.connected = True
                self.last_activity = time.time()
                self.error_count = 0
                
                # Uruchom timery
                self._start_keepalive()
                self._start_idle_timer()
                
                return True
            except Exception as e:
                self.last_error = str(e)
                self.error_count += 1
                self.connected = False
                raise
    
    def upload_file(self, file_path, remote_filename=None, max_retries=3):
        with self.connection_semaphore:
            if not remote_filename:
                remote_filename = os.path.basename(file_path)
                
            # Zmniejszamy opóźnienie między próbami
            time.sleep(random.uniform(0.05, 0.2))
            
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return {"success": False, "error": f"Plik nie istnieje lub jest pusty: {file_path}"}
            
            for attempt in range(max_retries):
                with self.lock:
                    if not self.connected and not self.connect():
                        if attempt == max_retries - 1:
                            return {"success": False, "error": f"Nie można połączyć się z FTP: {self.last_error}"}
                        time.sleep(min(FTP_CONFIG['MAX_RETRY_DELAY'], 
                                    FTP_CONFIG['DEFAULT_RETRY_DELAY'] * (attempt + 1)))
                        continue
                
                try:
                    with open(file_path, 'rb') as file:
                        with self.lock:
                            self.ftp.storbinary(f'STOR {remote_filename}', file)
                            self.last_activity = time.time()
                            self._start_idle_timer()  # Resetuj timer bezczynności

                    # Buduj URL
                    if self.settings.get("http_path"):
                        http_path = self.settings["http_path"].strip()
                        if not http_path.endswith('/'): http_path += '/'
                        image_url = f"{http_path}{remote_filename}"
                    else:
                        scheme = "ftps" if self.settings.get("use_tls") else "ftp"
                        image_url = f"{scheme}://{self.settings['host']}"
                        if self.settings["directory"] and self.settings["directory"] != "/":
                            if not self.settings["directory"].startswith("/"): image_url += "/"
                            image_url += self.settings["directory"]
                            if not image_url.endswith("/"): image_url += "/"
                        else:
                            image_url += "/"
                        image_url += remote_filename

                    return {"success": True, "url": image_url, "filename": remote_filename}
                except Exception as e:
                    print(f"Błąd wysyłania FTP (próba {attempt+1}): {str(e)}")
                    self.last_error = str(e)
                    self.error_count += 1
                    self.connected = False
                    
                    if attempt < max_retries - 1:
                        time.sleep(min(FTP_CONFIG['MAX_RETRY_DELAY'], 
                                    FTP_CONFIG['DEFAULT_RETRY_DELAY'] * (attempt + 1)))
                    
            return {"success": False, "error": f"Nie udało się wysłać pliku po {max_retries} próbach. Ostatni błąd: {self.last_error}"}
    
    def close(self):
        with self.lock:
            if self.keepalive_timer:
                self.keepalive_timer.cancel()
            if self.idle_timer:
                self.idle_timer.cancel()
            if self.connected and self.ftp:
                try: self.ftp.quit()
                except: pass
                self.connected = False
                
    def __del__(self):
        self.close()

class FTPConnectionPool:
    def __init__(self, max_connections=5):
        self.pool = ThreadPoolExecutor(max_workers=max_connections)
        self.connections = []
        self.lock = threading.Lock()
        
    def get_connection(self):
        with self.lock:
            if not self.connections:
                return FTPManager.get_instance(st.session_state.ftp_settings)
            return self.connections.pop()
            
    def release_connection(self, connection):
        with self.lock:
            self.connections.append(connection)
            
    def close_all(self):
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except:
                    pass
            self.connections.clear()

def calculate_batch_size(total_urls, max_workers):
    return min(
        max(FTP_CONFIG['MIN_BATCH_SIZE'], total_urls // 10),
        max_workers * 2
    )

class FTPBatchManager:
    """
    Manager do wykonywania operacji FTP w batchach,
    aby zmniejszyć obciążenie serwera FTP i uniknąć blokowania połączeń.
    """
    
    def __init__(self, settings, max_workers):
        self.settings = settings
        self.max_workers = max_workers
        self.batch_size = calculate_batch_size(max_workers * 10, max_workers)
        self.connection_pool = FTPConnectionPool(max_workers)
        self.upload_queue = queue.Queue()
        self.results = {}
        self.running = False
        self.worker_thread = None
        self.lock = threading.Lock()
        self.semaphore = threading.Semaphore(max_workers)
        self.ftp_manager = FTPManager.get_instance(settings)
        self.stats = {
            'total_uploaded': 0,
            'failed_uploads': 0,
            'retry_count': 0,
            'start_time': None,
            'last_upload_time': None
        }
        
    def verify_connection(self):
        """Weryfikuje połączenie FTP przed rozpoczęciem przetwarzania"""
        return self.ftp_manager.verify_connection()
        
    def add_upload_task(self, file_path, callback=None):
        """Dodaje zadanie do kolejki przesyłania"""
        task_id = str(uuid.uuid4())
        self.upload_queue.put((task_id, file_path, callback))
        self.results[task_id] = {"status": "queued", "file_path": file_path}
        return task_id
        
    def start_processing(self):
        """Rozpoczyna przetwarzanie zadań w osobnym wątku"""
        if self.running:
            return
            
        self.running = True
        self.stats['start_time'] = time.time()
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
    def stop_processing(self):
        """Zatrzymuje przetwarzanie zadań"""
        self.running = False
        if self.worker_thread:
            if self.worker_thread.is_alive():
                self.worker_thread.join(timeout=3)
            self.worker_thread = None
            
    def _process_queue(self):
        """Przetwarza zadania wysyłania w kolejce"""
        while self.running:
            try:
                batch = []
                for _ in range(self.batch_size):
                    if self.upload_queue.empty():
                        break
                    batch.append(self.upload_queue.get(block=False))
                
                if not batch:
                    time.sleep(0.1)  # Zmniejszone opóźnienie
                    continue
                
                time.sleep(random.uniform(0.1, 0.3))  # Zmniejszone opóźnienie między batchami
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(self._upload_file, task_id, file_path): (task_id, callback) 
                              for task_id, file_path, callback in batch}
                    
                    for future in concurrent.futures.as_completed(futures):
                        task_id, callback = futures[future]
                        try:
                            result = future.result()
                            self.results[task_id] = result
                            
                            # Aktualizuj statystyki
                            if result.get("status") == "success":
                                self.stats['total_uploaded'] += 1
                            elif result.get("status") == "error":
                                self.stats['failed_uploads'] += 1
                            self.stats['last_upload_time'] = time.time()
                            
                            if callback:
                                callback(task_id, result)
                        except Exception as e:
                            error_result = {
                                "status": "error", 
                                "error": f"Błąd wysyłania: {str(e)}",
                                "details": getattr(e, 'details', None)
                            }
                            self.results[task_id] = error_result
                            self.stats['failed_uploads'] += 1
                            if callback:
                                callback(task_id, error_result)
                
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"FTPBatchManager error: {str(e)}")
                time.sleep(0.5)
                
    def _upload_file(self, task_id, file_path):
        """Przesyła pojedynczy plik na FTP"""
        with self.semaphore:
            try:
                self.results[task_id] = {"status": "uploading"}
                upload_result = self.ftp_manager.upload_file(file_path)
                
                if upload_result["success"]:
                    return {
                        "status": "success", 
                        "url": upload_result["url"], 
                        "filename": upload_result["filename"]
                    }
                else:
                    self.stats['retry_count'] += 1
                    return {"status": "error", "error": upload_result["error"]}
            except Exception as e:
                self.stats['retry_count'] += 1
                return {"status": "error", "error": str(e)}
                
    def get_result(self, task_id):
        """Pobiera wynik dla zadania"""
        with self.lock:
            return self.results.get(task_id)
            
    def get_stats(self):
        """Zwraca statystyki przetwarzania"""
        with self.lock:
            stats = self.stats.copy()
            if stats['start_time']:
                stats['elapsed_time'] = time.time() - stats['start_time']
                if stats['total_uploaded'] > 0 and stats['elapsed_time'] > 0:
                    stats['upload_rate'] = stats['total_uploaded'] / stats['elapsed_time']
                else:
                    stats['upload_rate'] = 0
            return stats

# Funkcje pomocnicze
def authenticate_user():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Pobieranie zdjęć z XML/CSV - Logowanie")
        user = st.text_input("Login")
        password = st.text_input("Hasło", type="password")
        if st.button("Zaloguj"):
            if user == st.secrets.get("APP_USER") and password == st.secrets.get("APP_PASSWORD"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Nieprawidłowy login lub hasło")
        st.stop()
    return True

def initialize_session_state():
    defaults = {
        "generated_code": "", "edited_code": "", "output_bytes": None,
        "file_info": None, "show_editor": False, "error_info": None,
        "code_fixed": False, "fix_requested": False, "downloaded_images": [],
        "ftp_settings": {
            "host": "", "port": 21, "username": "", "password": "",
            "directory": "/", "http_path": ""
        },
        "processing_params": {
            "xpath": "", "column_name": "", "new_node_name": "",
            "new_column_name": "", "separator": ","
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def detect_encoding(raw_bytes):
    """Wykrywa kodowanie pliku"""
    # Sprawdź BOM
    if raw_bytes.startswith(codecs.BOM_UTF16_LE):
        return 'utf-16-le'
    if raw_bytes.startswith(codecs.BOM_UTF16_BE):
        return 'utf-16-be'
    if raw_bytes.startswith(codecs.BOM_UTF8):
        return 'utf-8'
    
    # Sprawdź deklarację XML
    encoding_match = re.search(br'<\?xml[^>]*encoding=["\']([^"\']+)["\']', raw_bytes)
    if encoding_match:
        try:
            return encoding_match.group(1).decode('ascii').lower()
        except:
            pass
    
    # Próbuj popularne kodowania
    for enc in ["utf-8", "iso-8859-2", "windows-1250"]:
        try:
            raw_bytes.decode(enc)
            return enc
        except:
            continue
    
    return 'utf-8'  # Domyślne kodowanie

def read_file_content(uploaded_file):
    if not uploaded_file:
        return None, "Nie wybrano pliku"
    
    try:
        raw_bytes = uploaded_file.read()
        file_type = uploaded_file.name.split(".")[-1].lower()
        
        if file_type not in ["xml", "csv"]:
            return None, "Nieobsługiwany typ pliku. Akceptowane formaty to XML i CSV."
        
        encoding = detect_encoding(raw_bytes)
        content = raw_bytes.decode(encoding)
        
        if file_type == "csv":
            # Weryfikacja czy to poprawny CSV
            try:
                pd.read_csv(io.StringIO(content))
            except:
                return None, "Niepoprawny format CSV"
        
        return {
            "content": content,
            "raw_bytes": raw_bytes,
            "type": file_type,
            "encoding": encoding,
            "name": uploaded_file.name
        }, None
        
    except Exception as e:
        return None, f"Błąd podczas odczytu pliku: {str(e)}"

def download_image(url, temp_dir, log_list=None):
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            if log_list is not None:
                log_list.append(f"ERROR: Nieprawidłowy URL: {url}")
            return None, f"Nieprawidłowy URL: {url}"

        headers = {
            "User-Agent": "Mozilla/5.0", "Accept": "*/*",
            "Referer": f"{parsed_url.scheme}://{parsed_url.netloc}/"
        }

        # Specjalna obsługa dla image_show.php
        if "image_show.php" in url:
            html_resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            html_resp.raise_for_status()
            soup = BeautifulSoup(html_resp.text, "html.parser")
            img_tag = soup.find("img")
            if not img_tag or not img_tag.get("src"):
                if log_list is not None:
                    log_list.append(f"WARNING: Brak <img> w HTML dla {url}")
                return None, "Nie znaleziono znacznika <img> w odpowiedzi HTML"
            img_src = img_tag["src"]
            img_url = f"{parsed_url.scheme}://{parsed_url.netloc}/{img_src.lstrip('/')}" if not img_src.startswith("http") else img_src
        else:
            img_url = url

        for retry in range(3):
            try:
                response = requests.get(img_url, headers=headers, stream=False, timeout=15, allow_redirects=True)
                if log_list is not None:
                    log_list.append(f"INFO: {img_url} -> HTTP {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")
                response.raise_for_status()
                extension = {
                    "image/jpeg": ".jpg", "image/png": ".png",
                    "image/gif": ".gif", "image/webp": ".webp"
                }.get(response.headers.get("Content-Type", ""), ".jpg")
                filename = f"image_{uuid.uuid4().hex}{extension}"
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(response.content)
                file_size = os.path.getsize(file_path)
                if log_list is not None:
                    log_list.append(f"INFO: Zapisano {file_path}, rozmiar: {file_size} bajtów")
                if file_size > 100 and file_size < 20 * 1024 * 1024:
                    return {"path": file_path, "filename": filename, "original_url": url}, None
                else:
                    os.remove(file_path)
                    if retry < 2:
                        continue
                    if file_size <= 100:
                        if log_list is not None:
                            log_list.append(f"WARNING: Plik za mały: {file_size} bajtów dla {url}")
                        return None, f"Pobrany plik jest zbyt mały (rozmiar: {file_size})"
                    else:
                        if log_list is not None:
                            log_list.append(f"WARNING: Plik za duży: {file_size} bajtów dla {url}")
                        return None, f"Plik jest zbyt duży (>20MB, rozmiar: {file_size})"
            except Exception as e:
                if log_list is not None:
                    log_list.append(f"ERROR: {img_url} -> {str(e)}")
                if retry == 2:
                    return None, f"Błąd przy pobieraniu: {str(e)}"
    except Exception as e:
        if log_list is not None:
            log_list.append(f"ERROR (outer): {url} -> {str(e)}")
        return None, f"Błąd: {str(e)}"

def process_single_url(url, retry_count=0, temp_dir=None, max_retries=3, log_list=None):
    """Pomocnicza funkcja do przetwarzania pojedynczego URL"""
    try:
        if log_list is not None:
            log_list.append(f"⬇️ Pobieranie: {url}")
        image_info, error = download_image(url, temp_dir, log_list=log_list)
        
        if error:
            # Dodane bardziej szczegółowe komunikaty dla częstych błędów
            if "image_show.php" in url and "Nie znaleziono znacznika" in error:
                specific_error = f"Problem z parsowaniem HTML: {error}"
            elif "timeout" in error.lower():
                specific_error = f"Timeout podczas pobierania: {error}"
            elif "connection" in error.lower():
                specific_error = f"Problem z połączeniem: {error}"
            else:
                specific_error = error
                
            if retry_count < max_retries:
                # Zwiększ czas oczekiwania między kolejnymi próbami
                wait_time = 1 + retry_count * 2  # 1s, 3s, 5s...
                if log_list is not None:
                    log_list.append(f"🔄 Ponawiam {retry_count + 1}/{max_retries} dla {url}: {specific_error} (czekam {wait_time}s)")
                time.sleep(wait_time)
                return {
                    "status": "retry",
                    "url": url,
                    "retry_count": retry_count + 1,
                    "error": specific_error
                }
            else:
                if log_list is not None:
                    log_list.append(f"❌ Błąd pobierania dla {url}: {specific_error}")
                return {
                    "status": "error",
                    "url": url,
                    "error": specific_error
                }
        
        if not image_info or not image_info.get("path"):
            return {
                "status": "error",
                "url": url,
                "error": "Brak informacji o pobranym pliku"
            }
            
        if log_list is not None:
            log_list.append(f"✅ Pobrano: {url}")
            
        return {
            "status": "success",
            "url": url,
            "image_info": image_info
        }
        
    except Exception as e:
        # Poprawiona obsługa wyjątków
        exception_details = f"{type(e).__name__}: {str(e)}"
        if retry_count < max_retries:
            wait_time = 1 + retry_count * 2
            if log_list is not None:
                log_list.append(f"🔄 Ponawiam {retry_count + 1}/{max_retries} dla {url}: {exception_details} (czekam {wait_time}s)")
            time.sleep(wait_time)
            return {
                "status": "retry",
                "url": url,
                "retry_count": retry_count + 1,
                "error": exception_details
            }
        else:
            if log_list is not None:
                log_list.append(f"⛔ Błąd dla {url}: {exception_details}")
            return {
                "status": "error",
                "url": url,
                "error": exception_details
            }

def process_images_in_parallel(urls, temp_dir, ftp_settings, max_workers=None, debug_container=None, max_retries=3, progress_callback=None, session_id=None):
    # Użyj domyślnej wartości jeśli nie podano max_workers
    if max_workers is None:
        max_workers = FTP_CONFIG['DEFAULT_WORKERS']
    
    # Upewnij się, że max_workers jest w dozwolonym zakresie
    max_workers = max(FTP_CONFIG['MIN_WORKERS'], 
                     min(FTP_CONFIG['MAX_WORKERS'], max_workers))
    
    # Inicjalizacja FTP Batch Managera z odpowiednimi parametrami
    ftp_batch_manager = FTPBatchManager(ftp_settings, max_workers)
    
    if debug_container:
        debug_container.info(f"Konfiguracja przetwarzania:")
        debug_container.info(f"- Max równoległych procesów: {max_workers}")
        debug_container.info(f"- Rozmiar batcha: {ftp_batch_manager.batch_size}")
        debug_container.info(f"- Max połączeń FTP: {ftp_batch_manager.max_workers}")
    
    # Weryfikacja połączenia FTP przed rozpoczęciem
    success, message = ftp_batch_manager.verify_connection()
    if not success:
        if debug_container:
            debug_container.error(f"❌ Błąd połączenia FTP: {message}")
        return {}, [], [{"url": "all", "error": f"Błąd połączenia FTP: {message}"}]
    
    if debug_container:
        debug_container.success("✅ Połączenie FTP zweryfikowane pomyślnie")
    
    new_urls_map = {}
    downloaded_images = []
    failed_urls = []
    retry_queue = queue.Queue()
    
    # Initialize counters
    total_urls = len(urls)
    processed_count = 0
    
    # Inicjalizacja postępu - pokaż 0% na starcie
    if progress_callback:
        progress_callback(0, 0, 0, total_urls)
        
    if debug_container:
        debug_container.info(f"Inicjalizacja przetwarzania dla {total_urls} obrazów...")
    
    # Inicjalizuj FTP Batch Manager
    ftp_batch_manager.start_processing()
    
    # Słownik do śledzenia zadań FTP
    ftp_tasks = {}
    ftp_results_lock = threading.Lock()
    
    def check_ftp_tasks():
        """Sprawdza status zadań FTP i aktualizuje wyniki"""
        nonlocal processed_count, new_urls_map
        completed_tasks = []
        
        with ftp_results_lock:
            for url, task_info in ftp_tasks.items():
                result = ftp_batch_manager.get_result(task_info["task_id"])
                if not result:
                    # Zadanie jeszcze nie zakończone
                    continue
                    
                if result["status"] == "success":
                    new_urls_map[url] = result["url"]
                    downloaded_images.append({
                        "original_url": url,
                        "ftp_url": result["url"],
                        "filename": result["filename"]
                    })
                    completed_tasks.append(url)
                    processed_count += 1
                    if debug_container:
                        debug_container.success(f"✅ FTP zakończone: {url}")
                elif result["status"] == "error":
                    # Bardziej szczegółowa diagnostyka błędów FTP
                    error_details = result.get("error", "Nieznany błąd FTP")
                    if "connection" in error_details.lower():
                        error_category = "Błąd połączenia FTP"
                    elif "permission" in error_details.lower() or "access" in error_details.lower():
                        error_category = "Błąd uprawnień FTP"
                    elif "disk" in error_details.lower() or "space" in error_details.lower():
                        error_category = "Błąd przestrzeni dyskowej"
                    else:
                        error_category = "Błąd FTP"
                        
                    if task_info["retry_count"] < max_retries:
                        new_retry_count = task_info["retry_count"] + 1
                        # Zwiększ czas oczekiwania między próbami
                        wait_time = min(5, 1 * (new_retry_count))
                        time.sleep(wait_time)
                        
                        retry_queue.put((url, new_retry_count))
                        completed_tasks.append(url)
                        
                        if debug_container:
                            debug_container.warning(f"🔄 Ponawiam FTP {new_retry_count}/{max_retries} dla {url}: {error_details}")
                    else:
                        failed_urls.append({"url": url, "error": f"{error_category}: {error_details}"})
                        completed_tasks.append(url)
                        processed_count += 1
                        if debug_container:
                            debug_container.error(f"❌ FTP nieudane: {url}: {error_details}")
            
            # Usuń zakończone zadania
            for url in completed_tasks:
                if url in ftp_tasks:
                    del ftp_tasks[url]
    
    # Dodajemy tylko początkowy batch URL-i
    initial_batch_size = min(max_workers * 2, total_urls)
    for url in urls[:initial_batch_size]:
        retry_queue.put((url, 0))
    
    next_url_index = initial_batch_size
    
    # Process until queue is empty and all URLs are processed
    log_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while not retry_queue.empty() or next_url_index < total_urls or ftp_tasks:
            # Sprawdź status zadań FTP
            check_ftp_tasks()
            
            # Dodajemy więcej URL-i do kolejki w miarę potrzeby
            while next_url_index < total_urls and retry_queue.qsize() < max_workers * 2:
                retry_queue.put((urls[next_url_index], 0))
                next_url_index += 1
                
            # Informacja o stanie kolejki
            queue_size = retry_queue.qsize()
            if debug_container and (next_url_index < total_urls or queue_size > 0 or ftp_tasks):
                ftp_queue_size = len(ftp_tasks)
                debug_container.info(f"📋 W kolejce pobierania: {queue_size} | W kolejce FTP: {ftp_queue_size} | Zaplanowano: {next_url_index}/{total_urls}")
            
            if retry_queue.empty() and next_url_index >= total_urls and ftp_tasks:
                time.sleep(0.1)
                continue
            
            # Get batch of URLs from queue
            batch = []
            batch_size = min(max_workers, queue_size)
            for _ in range(batch_size):
                if not retry_queue.empty():
                    batch.append(retry_queue.get())
            
            if not batch:
                if next_url_index < total_urls:
                    continue
                elif ftp_tasks:
                    time.sleep(0.1)
                    continue
                else:
                    break
            
            # Process batch
            futures = {
                executor.submit(
                    process_single_url,
                    url=url,
                    retry_count=retry_count,
                    temp_dir=temp_dir,
                    max_retries=max_retries,
                    log_list=log_list
                ): (url, retry_count) for url, retry_count in batch
            }
            
            # --- Zmienna do liczenia ile zdjęć od ostatniej aktualizacji ---
            images_since_update = 0
            batch_size_total = len(futures)
            batch_processed = 0
            # ---
            for future in concurrent.futures.as_completed(futures):
                url, retry_count = futures[future]
                try:
                    result = future.result()
                    
                    if result["status"] == "success":
                        # Dodaj zadanie do kolejki FTP
                        task_id = ftp_batch_manager.add_upload_task(result["image_info"]["path"])
                        with ftp_results_lock:
                            ftp_tasks[url] = {
                                "task_id": task_id,
                                "file_path": result["image_info"]["path"],
                                "retry_count": retry_count
                            }
                        if debug_container:
                            debug_container.info(f"⬆️ Zadanie FTP dodane: {url}")
                    elif result["status"] == "retry":
                        if debug_container:
                            debug_container.info(f"🔄 Ponawiam {result['retry_count']}/{max_retries} dla {url}: {result.get('error')}")
                        time.sleep(random.uniform(0.5, 1.0))
                        retry_queue.put((url, result["retry_count"]))
                    else:  # error
                        failed_urls.append({"url": url, "error": result.get("error", "Nieznany błąd")})
                        processed_count += 1
                        if debug_container:
                            debug_container.warning(f"❌ Błąd dla {url}: {result.get('error')}")

                    # --- Aktualizacja postępu co 10 zdjęć lub na końcu batcha ---
                    batch_processed += 1
                    images_since_update += 1
                    if progress_callback and total_urls > 0:
                        if images_since_update >= 10 or batch_processed == batch_size_total:
                            remaining = total_urls - next_url_index + retry_queue.qsize() + len(ftp_tasks)
                            progress = min(0.99, max(0.01, (total_urls - remaining) / total_urls))
                            stats = ftp_batch_manager.get_stats()
                            progress_callback(progress, processed_count, stats['total_uploaded'], total_urls)
                            images_since_update = 0
                            save_current_state()  # Zapisz stan co 10 obrazów
                    # ---
                except Exception as e:
                    failed_urls.append({"url": url, "error": str(e)})
                    processed_count += 1
                    if debug_container:
                        debug_container.error(f"⛔ Wyjątek dla {url}: {str(e)}")
    
    # Zatrzymaj FTP Batch Manager
    ftp_batch_manager.stop_processing()
    
    # Final progress update
    if progress_callback and total_urls > 0:
        progress_callback(1.0, processed_count, processed_count, total_urls)
    
    # Close all FTP connections
    for instance in FTPManager._instances.values():
        instance.close()
    
    # Po zakończeniu batcha wyświetl logi
    if log_list:
        with st.expander("Logi debugowania (pobieranie obrazów)"):
            for log in log_list:
                st.write(log)
    
    return new_urls_map, downloaded_images, failed_urls

def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return None, f"Błąd: {str(e)}"
    return wrapper

@handle_errors
def extract_image_urls_from_xml(xml_content, xpath_expression, separator=","):
    if not xml_content or not xml_content.strip():
        return None, "Plik XML jest pusty"
    
    # Oczyść dane wejściowe
    if xml_content.startswith("\ufeff"):
        xml_content = xml_content[1:]
    xml_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_content)
    
    # Obsługa atrybutów
    is_attribute = '/@' in xpath_expression
    attribute_name = xpath_expression.split('/@')[-1] if is_attribute else None
    xpath_base = xpath_expression.split('/@')[0] if is_attribute else xpath_expression

    # Wyrażenia regularne dla najczęstszych przypadków
    if xpath_base in ["//product/image", "product/image", "//image", "/image"]:
        try:
            # Ulepszone wyrażenia regularne dla różnych formatów
            pattern_simple = re.compile(r'<image>(.*?)</image>', re.DOTALL)
            pattern_cdata = re.compile(r'<image><!\[CDATA\[(.*?)\]\]></image>', re.DOTALL)
            pattern_with_attribs = re.compile(r'<image [^>]*?>(.*?)</image>', re.DOTALL)
            
            matches = pattern_simple.findall(xml_content) + pattern_cdata.findall(xml_content) + pattern_with_attribs.findall(xml_content)
            
            urls = []
            for match in matches:
                match = match.strip()
                if not match:
                    continue
                
                # Znajdź URL-e w treści
                url_pattern = re.compile(r'https?://[^\s<>"\']+')
                found_urls = url_pattern.findall(match)
                
                if found_urls:
                    for url in found_urls:
                        clean_url = url.replace('&amp;', '&')
                        urls.append(clean_url)
                elif 'http://' in match or 'https://' in match:
                    urls.append(match.replace('&amp;', '&'))
            
            if urls:
                return urls, None
        except Exception as e:
            return None, f"Błąd przy parsowaniu XML wyrażeniami regularnymi: {str(e)}"

    # ElementTree dla innych przypadków
    try:
        # Próbuj naprawić częste problemy w XML
        xml_content = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;)', '&amp;', xml_content)
        
        root = ET.fromstring(xml_content)
        xpath = f"./{xpath_base[2:]}" if xpath_base.startswith('//') else f"./{xpath_base}" if not xpath_base.startswith('./') else xpath_base
        elements = root.findall(xpath)

        urls = []
        for element in elements:
            element_text = element.attrib.get(attribute_name) if is_attribute else element.text
            if element_text:
                element_text = element_text.replace('&amp;', '&')
                
                # Sprawdź, czy tekst zawiera URL-e
                if 'http://' in element_text or 'https://' in element_text:
                    if separator in element_text:
                        found_urls = [url.strip() for url in element_text.split(separator) 
                                    if url.strip() and ('http://' in url or 'https://' in url)]
                        urls.extend(found_urls)
                    else:
                        urls.append(element_text.strip())
        
        # Jeśli nie znaleziono URL-i, spróbuj bardziej generyczne podejście
        if not urls:
            # Znajdź wszystkie elementy z tekstem zawierającym http
            for elem in root.iter():
                if elem.text and ('http://' in elem.text or 'https://' in elem.text):
                    element_text = elem.text.replace('&amp;', '&')
                    if separator in element_text:
                        found_urls = [url.strip() for url in element_text.split(separator) 
                                    if url.strip() and ('http://' in url or 'https://' in url)]
                        urls.extend(found_urls)
                    else:
                        urls.append(element_text.strip())
        
        return urls, None
    except ET.ParseError as e:
        return None, f"Błąd przy parsowaniu XML: {str(e)}"

def update_xml_with_new_urls(xml_content, xpath_expression, new_urls_map, new_node_name, separator=","):
    try:
        if not xml_content.strip() or not new_node_name.strip():
            return None, "Plik XML jest pusty lub nazwa węzła jest pusta"

        # Obsługa atrybutów
        is_attribute = '/@' in xpath_expression
        attribute_name = xpath_expression.split('/@')[-1] if is_attribute else None
        xpath_base = xpath_expression.split('/@')[0] if is_attribute else xpath_expression

        root = ET.fromstring(xml_content)
        xpath = xpath_base[2:] if xpath_base.startswith('//') else xpath_base
        elements = root.findall(f'.//{xpath}')
        
        # Mapa rodzic-dziecko dla szybkiego odnajdywania
        parent_map = {c: p for p in root.iter() for c in p}
        
        # Mapa rodzic-elementy do przetworzenia
        parent_to_elements = {}
        for element in elements:
            parent = parent_map.get(element)
            if parent is None:
                continue
            if parent not in parent_to_elements:
                parent_to_elements[parent] = []
            parent_to_elements[parent].append(element)
        
        # Przetwarzanie elementów
        for parent, elements_list in parent_to_elements.items():
            # Znajdź lub utwórz węzeł ftp_images
            ftp_images = None
            for child in parent:
                if child.tag == "ftp_images":
                    ftp_images = child
                    break
            if ftp_images is None:
                ftp_images = ET.Element("ftp_images")
                parent.append(ftp_images)
            
            # Przetwarzanie elementów
            for element in elements_list:
                # Pobierz oryginalny URL
                original_url = element.attrib.get(attribute_name, "").strip() if is_attribute else (element.text.strip() if element.text else "")
                if not original_url:
                    continue
                
                # Obsługa wielu URL-i w jednym elemencie
                if separator in original_url:
                    urls = [url.strip() for url in original_url.split(separator)]
                    new_urls = [new_urls_map[url] for url in urls if url in new_urls_map]
                    if new_urls:
                        ftp_node = ET.Element(new_node_name)
                        ftp_node.text = separator.join(new_urls)
                        ftp_images.append(ftp_node)
                # Obsługa pojedynczego URL-a
                elif original_url in new_urls_map:
                    ftp_node = ET.Element(new_node_name)
                    ftp_node.text = new_urls_map[original_url]
                    ftp_images.append(ftp_node)

        return ET.tostring(root, encoding="unicode"), None
    except Exception as e:
        return None, f"Błąd przy aktualizacji XML: {str(e)}"

@handle_errors
def extract_image_urls_from_csv(csv_content, column_name, separator=","):
    df = pd.read_csv(io.StringIO(csv_content))
    if column_name not in df.columns:
        return None, f"Kolumna '{column_name}' nie istnieje w pliku CSV."
    
    urls = []
    for value in df[column_name]:
        if pd.notna(value):
            if separator in str(value):
                urls.extend([url.strip() for url in str(value).split(separator) if url.strip()])
            else:
                urls.append(str(value).strip())
    
    return urls, None

def update_csv_with_new_urls(csv_content, column_name, new_urls_map, new_column_name, separator=","):
    try:
        if not new_column_name.strip():
            return None, "Nazwa nowej kolumny nie może być pusta"

        df = pd.read_csv(io.StringIO(csv_content))
        if column_name not in df.columns:
            return None, f"Kolumna '{column_name}' nie istnieje w pliku CSV."

        if new_column_name not in df.columns:
            df[new_column_name] = ""

        for idx, value in enumerate(df[column_name]):
            if pd.notna(value):
                value_str = str(value).strip()
                
                # Obsługa wielu URL-i
                if separator in value_str:
                    urls = [url.strip() for url in value_str.split(separator)]
                    new_urls = [new_urls_map[url] for url in urls if url in new_urls_map]
                    if new_urls:
                        df.at[idx, new_column_name] = separator.join(new_urls)
                # Obsługa pojedynczego URL-a
                elif value_str in new_urls_map:
                    df.at[idx, new_column_name] = new_urls_map[value_str]
                else:
                    # Próba dopasowania bez białych znaków
                    for key in new_urls_map:
                        if value_str.replace(" ", "") == key.replace(" ", ""):
                            df.at[idx, new_column_name] = new_urls_map[key]
                            break

        return df.to_csv(index=False), None
    except Exception as e:
        return None, f"Błąd przy aktualizacji CSV: {str(e)}"

def save_to_google_drive(output_bytes, file_info, new_urls_map=None):
    try:
        drive_folder_id = st.secrets.get("GOOGLE_DRIVE_FOLDER_ID")
        credentials_json = st.secrets.get("GOOGLE_DRIVE_CREDENTIALS_JSON")
        
        if not drive_folder_id or not credentials_json:
            return False, "Brak konfiguracji Google Drive."

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"image_urls_map_{now}.txt"
        result_filename = f"processed_{now}.{file_info['type']}"
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Zapisz pliki tymczasowo
            temp_result_path = os.path.join(tmpdirname, f"output.{file_info['type']}")
            temp_log_path = os.path.join(tmpdirname, "log.txt")
            
            with open(temp_result_path, "wb") as f:
                f.write(output_bytes)
                
            # Przygotuj log
            log_content = f"# Raport z przetwarzania obrazów - {now}\n\n## Informacje o pliku\n"
            log_content += f"- Nazwa pliku: {file_info['name']}\n- Typ pliku: {file_info['type'].upper()}\n"
            log_content += f"- Kodowanie: {file_info['encoding']}\n\n## Mapowanie URL-i obrazów\n\n"
            
            if new_urls_map:
                for i, (original_url, new_url) in enumerate(new_urls_map.items(), 1):
                    log_content += f"### Obraz #{i}\n- Oryginalny URL: {original_url}\n- Nowy URL: {new_url}\n\n"
            else:
                log_content += "Brak mapowania URL-i\n"
                
            with open(temp_log_path, "w", encoding='utf-8') as f:
                f.write(log_content)
            
            # Uwierzytelnianie Google Drive
            with st.spinner("Zapisuję na Google Drive..."):
                # Przygotuj poświadczenia
                if isinstance(credentials_json, str):
                    try:
                        creds_dict = json.loads(credentials_json)
                    except:
                        return False, "Błąd dekodowania JSON z credentials"
                else:
                    creds_dict = credentials_json
                
                scope = ["https://www.googleapis.com/auth/drive"]
                credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
                
                gauth = GoogleAuth()
                gauth.credentials = credentials
                drive = GoogleDrive(gauth)
                
                try:
                    # Prześlij pliki
                    log_file = drive.CreateFile({
                        "title": log_filename, 
                        "parents": [{"id": drive_folder_id}],
                        "mimeType": "text/plain"
                    })
                    log_file.SetContentFile(temp_log_path)
                    log_file.Upload()
                    
                    result_file = drive.CreateFile({
                        "title": result_filename, 
                        "parents": [{"id": drive_folder_id}],
                        "mimeType": f"application/{file_info['type']}"
                    })
                    result_file.SetContentFile(temp_result_path)
                    result_file.Upload()
                    
                    return True, "Pliki zostały zapisane na Google Drive."
                except Exception as e:
                    return False, f"Błąd podczas wysyłania: {str(e)}"
    except Exception as e:
        return False, f"Błąd Google Drive: {str(e)}"

# Funkcje zarządzania stanem
def save_processing_state(session_id, urls, processed_urls, new_urls_map, file_info, processing_params):
    # Ensure all lists have unique entries
    all_urls = list(set(urls))
    processed_urls = list(set(processed_urls))
    
    # Calculate remaining URLs accurately
    remaining_urls = [url for url in all_urls if url not in processed_urls]
    
    state = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "file_info": {k: file_info[k] for k in ["name", "type", "encoding"]},
        "processing_params": processing_params,
        "total_urls": len(all_urls),
        "processed_urls": processed_urls,
        "new_urls_map": new_urls_map,
        "remaining_urls": remaining_urls,
        "file_content": file_info.get("content", "")
    }
    
    state_dir = os.path.join(os.path.expanduser("~"), ".xml_image_processor")
    os.makedirs(state_dir, exist_ok=True)
    state_file = os.path.join(state_dir, f"session_{session_id}.json")
    
    with open(state_file, "w") as f:
        json.dump(state, f)
    
    return state_file

def verify_state_consistency(state):
    """Weryfikuje spójność stanu i zwraca listę problemów"""
    problems = []
    
    required_keys = ["session_id", "timestamp", "file_info", "processing_params", 
                    "total_urls", "processed_urls", "new_urls_map", "remaining_urls"]
    
    # Sprawdź wymagane klucze
    for key in required_keys:
        if key not in state:
            problems.append(f"Brak wymaganego klucza: {key}")
            
    # Sprawdź spójność liczby URL-i
    if "total_urls" in state and "processed_urls" in state and "remaining_urls" in state:
        total = state["total_urls"]
        processed = len(state["processed_urls"])
        remaining = len(state["remaining_urls"])
        
        if total != processed + remaining:
            problems.append(f"Niespójność liczby URL-i: total={total}, processed={processed}, remaining={remaining}")
            
    # Sprawdź spójność mapowania URL-i
    if "new_urls_map" in state and "processed_urls" in state:
        mapped_urls = set(state["new_urls_map"].keys())
        processed_urls = set(state["processed_urls"])
        
        if mapped_urls != processed_urls:
            problems.append("Niespójność między przetworzonymi URL-ami a mapowaniem")
            
    return problems

def load_processing_state(session_id=None):
    state_dir = os.path.join(os.path.expanduser("~"), ".xml_image_processor")
    if not os.path.exists(state_dir):
        return None
    
    if session_id:
        state_file = os.path.join(state_dir, f"session_{session_id}.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                    
                # Sprawdź spójność stanu
                problems = verify_state_consistency(state)
                if problems:
                    st.warning("⚠️ Wykryto problemy ze stanem sesji:")
                    for problem in problems:
                        st.warning(f"- {problem}")
                    
                    # Spróbuj wczytać kopię zapasową
                    backup_file = state_file + ".bak"
                    if os.path.exists(backup_file):
                        with open(backup_file, "r") as f:
                            backup_state = json.load(f)
                        backup_problems = verify_state_consistency(backup_state)
                        if not backup_problems:
                            st.info("✅ Wczytano kopię zapasową stanu")
                            return backup_state
                            
                return state
            except Exception as e:
                st.error(f"Błąd podczas wczytywania stanu: {str(e)}")
                return None
        return None
    
    # Znajdź najnowszy plik
    state_files = [f for f in os.listdir(state_dir) if f.startswith("session_") and f.endswith(".json")]
    if not state_files:
        return None
    
    state_files.sort(key=lambda x: os.path.getmtime(os.path.join(state_dir, x)), reverse=True)
    with open(os.path.join(state_dir, state_files[0]), "r") as f:
        state = json.load(f)
        
    # Sprawdź spójność najnowszego stanu
    problems = verify_state_consistency(state)
    if problems:
        st.warning("⚠️ Wykryto problemy z najnowszym stanem:")
        for problem in problems:
            st.warning(f"- {problem}")
    
    return state

def list_saved_sessions():
    state_dir = os.path.join(os.path.expanduser("~"), ".xml_image_processor")
    if not os.path.exists(state_dir):
        return []
    
    sessions = []
    for filename in os.listdir(state_dir):
        if filename.startswith("session_") and filename.endswith(".json"):
            try:
                with open(os.path.join(state_dir, filename), "r") as f:
                    state = json.load(f)
                    progress = round(len(state["processed_urls"]) / state["total_urls"] * 100, 1) if state["total_urls"] > 0 else 0
                    sessions.append({
                        "session_id": state["session_id"],
                        "timestamp": state["timestamp"],
                        "file_info": state["file_info"]["name"],
                        "progress": f"{len(state['processed_urls'])}/{state['total_urls']}",
                        "percentage": progress
                    })
            except:
                pass
    
    sessions.sort(key=lambda x: x["timestamp"], reverse=True)
    return sessions

def resume_processing(state, temp_dir, ftp_settings, max_workers=5):
    remaining_urls = state["remaining_urls"]
    new_urls_map = state.get("new_urls_map", {})
    file_info = state["file_info"]
    processing_params = state["processing_params"]
    session_id = state["session_id"]
    
    # Przywróć parametry przetwarzania do sesji
    st.session_state.file_info = file_info
    st.session_state.processing_params = processing_params
    
    if not remaining_urls:
        st.success("Wszystkie URL-e zostały już przetworzone.")
        return True
    
    progress_bar = st.progress(len(state["processed_urls"]) / state["total_urls"])
    status_text = st.empty()
    debug_area = st.empty()
    
    total_to_process = len(remaining_urls)
    status_text.text(f"Inicjalizacja wznawiania przetwarzania {len(remaining_urls)} pozostałych obrazów...")
    
    # Pre-inicjalizacja FTP dla szybszego startu
    if debug_area:
        debug_area.info("Przygotowanie połączenia FTP...")
    try:
        ftp_manager = FTPManager.get_instance(ftp_settings)
        ftp_manager.connect()
    except Exception as e:
        if debug_area:
            debug_area.warning(f"Problem z wstępnym połączeniem FTP: {str(e)}")
            
    # Mierzenie czasu wykonania
    start_time = time.time()
    
    # Funkcja aktualizacji postępu z większą ilością szczegółów
    def update_progress(progress_value, processed, total_processed, total):
        # Oblicz całkowity progres uwzględniając już przetworzone obrazy
        overall_progress = (len(state["processed_urls"]) + total_processed) / state["total_urls"]
        progress_bar.progress(overall_progress)
        
        percent = int(progress_value*100)
        elapsed_time = time.time() - start_time
        
        if processed > 0 and elapsed_time > 0:
            speed = processed / elapsed_time
            remaining = (total - processed) / speed if speed > 0 else 0
            time_info = f" | ~{int(remaining/60)}m {int(remaining%60)}s pozostało"
            speed_info = f" | {speed:.1f} obrazów/s"
        else:
            time_info = ""
            speed_info = ""
            
        status_text.text(f"Przetwarzanie... {total_processed}/{total} ({percent}%){speed_info}{time_info}")
    
    # Use our improved parallel processing function
    batch_result, batch_downloaded, failed_urls = process_images_in_parallel(
        remaining_urls, 
        temp_dir, 
        ftp_settings,
        max_workers=max_workers,
        debug_container=debug_area,
        max_retries=3,
        progress_callback=update_progress,
        session_id=session_id
    )
    
    # Update state with new results
    new_urls_map.update(batch_result)
    processed_urls = state.get("processed_urls", []) + list(batch_result.keys())
    
    # Remove duplicates if any
    processed_urls = list(set(processed_urls))
    
    # Save the updated state
    save_processing_state(
        session_id, 
        list(set(state["remaining_urls"] + state["processed_urls"])),  # Remove duplicates 
        processed_urls, 
        new_urls_map, 
        file_info,
        processing_params
    )
    
    status_text.text(f"Zakończono wznowione przetwarzanie. Pobrano i przesłano {len(batch_downloaded)} obrazów.")
    
    # Report on failed URLs
    if failed_urls:
        with st.expander(f"Nie udało się przetworzyć {len(failed_urls)} URL-i"):
            for fail in failed_urls[:20]:
                st.warning(f"{fail['url']}: {fail['error']}")
            if len(failed_urls) > 20:
                st.warning(f"... oraz {len(failed_urls) - 20} więcej.")
    
    # Aktualizacja pliku po wszystkich pobraniach
    if new_urls_map:
        file_type = file_info["type"]
        file_content = state.get("file_content", "")
        
        if not file_content:
            st.error("Brak treści pliku w zapisanym stanie - nie można zaktualizować pliku.")
            return False
        
        # Aktualizacja odpowiedniego typu pliku
        if file_type == "xml":
            xpath = processing_params.get("xpath", "")
            new_node_name = processing_params.get("new_node_name", "ftp")
            separator = processing_params.get("separator", ",")
            
            updated_content, error = update_xml_with_new_urls(
                file_content, xpath, new_urls_map, new_node_name, separator
            )
        else:
            column_name = processing_params.get("column_name", "")
            new_column_name = processing_params.get("new_column_name", "ftp_url")
            separator = processing_params.get("separator", ",")
            
            updated_content, error = update_csv_with_new_urls(
                file_content, column_name, new_urls_map, new_column_name, separator
            )
        
        if error:
            st.error(f"Błąd podczas aktualizacji pliku: {error}")
            return False
        
        output_bytes = updated_content.encode(file_info["encoding"])
        st.success("Plik został pomyślnie zaktualizowany!")
        
        # Google Drive
        try:
            success, message = save_to_google_drive(output_bytes, file_info, new_urls_map)
            st.success(f"✅ {message}") if success else st.warning(f"⚠️ {message}")
        except Exception as e:
            st.error(f"Błąd Google Drive: {str(e)}")
        
        # Przycisk pobierania
        base_name = os.path.splitext(file_info["name"])[0]
        st.download_button(
            label=f"📁 Pobierz zaktualizowany plik",
            data=output_bytes,
            file_name=f"{base_name}_updated.{file_type}",
            mime="text/plain"
        )
        return True
    else:
        st.warning("Nie udało się przetworzyć żadnych nowych obrazów.")
        return False

def reset_app_state():
    for key in list(st.session_state.keys()):
        if key not in ["authenticated", "ftp_settings"]:
            del st.session_state[key]
    initialize_session_state()
    st.rerun()

def main():
    st.set_page_config(page_title="Pobieranie zdjęć z XML/CSV", layout="centered")
    authenticate_user()
    initialize_session_state()

    st.title("Pobieranie zdjęć z XML/CSV")

    tab1, tab2, tab3 = st.tabs(["Pobieranie zdjęć", "Wznów przetwarzanie", "Pomoc"])

    with tab1:
        st.markdown("""
        To narzędzie umożliwia pobieranie zdjęć z plików XML lub CSV i zapisywanie ich na serwerze FTP.
        Prześlij plik, wskaż lokalizację linków do zdjęć, podaj dane FTP i pobierz zdjęcia.
        """)

        st.subheader("1. Wczytaj plik źródłowy")
        uploaded_file = st.file_uploader("Wgraj plik XML lub CSV", type=["xml", "csv"])

        if uploaded_file:
            file_info, error = read_file_content(uploaded_file)
            if error:
                st.error(error)
            else:
                st.success(f"Wczytano plik: {file_info['name']} ({file_info['type'].upper()}, {file_info['encoding']})")
                st.session_state.file_info = file_info

        st.subheader("2. Konfiguracja pobierania zdjęć")

        if st.session_state.file_info:
            file_type = st.session_state.file_info["type"]

            if file_type == "xml":
                xpath = st.text_input("XPath do węzła zawierającego URL-e zdjęć", 
                                    placeholder="Np. //product/image lub //image/@url")
                new_node_name = st.text_input("Nazwa nowego węzła dla linków FTP", 
                                            placeholder="Np. ftp")
                st.session_state.processing_params["xpath"] = xpath
                st.session_state.processing_params["new_node_name"] = new_node_name
            else:
                column_name = st.text_input("Nazwa kolumny zawierającej URL-e zdjęć", 
                                          placeholder="Np. image_url")
                new_column_name = st.text_input("Nazwa nowej kolumny dla linków FTP", 
                                              placeholder="Np. ftp_image_url")
                st.session_state.processing_params["column_name"] = column_name
                st.session_state.processing_params["new_column_name"] = new_column_name

            separator = st.text_input("Separator URL-i (jeśli w jednej komórce/węźle jest wiele linków)", value=",")
            st.session_state.processing_params["separator"] = separator

        st.subheader("3. Konfiguracja serwera FTP")

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.ftp_settings["host"] = st.text_input("Adres serwera FTP", 
                                                                 value=st.session_state.ftp_settings["host"])
            st.session_state.ftp_settings["port"] = st.number_input("Port", value=st.session_state.ftp_settings["port"],
                                                                   min_value=1, max_value=65535)
            st.session_state.ftp_settings["directory"] = st.text_input("Katalog docelowy", 
                                                                      value=st.session_state.ftp_settings["directory"])
            st.session_state.ftp_settings["http_path"] = st.text_input("Ścieżka HTTP do zdjęć",
                                                                      value=st.session_state.ftp_settings.get("http_path", ""),
                                                                      placeholder="https://example.com/images/")

        with col2:
            st.session_state.ftp_settings["username"] = st.text_input("Nazwa użytkownika", 
                                                                     value=st.session_state.ftp_settings["username"])
            st.session_state.ftp_settings["password"] = st.text_input("Hasło", type="password", 
                                                                     value=st.session_state.ftp_settings["password"])

        st.subheader("4. Pobierz zdjęcia i prześlij na FTP")
        
        if st.session_state.file_info:
            max_workers = st.slider(
                "Liczba równoległych procesów pobierania", 
                min_value=FTP_CONFIG['MIN_WORKERS'],
                max_value=FTP_CONFIG['MAX_WORKERS'],
                value=FTP_CONFIG['DEFAULT_WORKERS'],
                help="Wyższa wartość przyspieszy pobieranie, ale może obciążyć łącze"
            )

        # --- Nowa sekcja: kontenery na status i kolejkę ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        queue_info = st.empty()
        debug_area = st.empty()
        # ---

        if st.session_state.file_info and st.button("Pobierz zdjęcia i prześlij na FTP"):
            try:
                # Generujemy nowe ID sesji
                session_id = f"{uuid.uuid4().hex}_{int(time.time())}"
                
                # Walidacja wejść
                file_type = st.session_state.file_info["type"]
                file_content = st.session_state.file_info["content"]
                
                if file_type == "xml" and (not xpath or not xpath.strip() or not new_node_name or not new_node_name.strip()):
                    st.error("Podaj prawidłowy XPath i nazwę nowego węzła!")
                    return
                elif file_type == "csv" and (not column_name or not column_name.strip() or not new_column_name or not new_column_name.strip()):
                    st.error("Podaj prawidłową nazwę kolumny i nazwę nowej kolumny!")
                    return
                    
                # Ekstrakcja URL-i
                if file_type == "xml" and xpath:
                    urls, error = extract_image_urls_from_xml(file_content, xpath, separator)
                elif file_type == "csv" and column_name:
                    urls, error = extract_image_urls_from_csv(file_content, column_name, separator)
                else:
                    urls, error = None, "Nie podano ścieżki XPath lub nazwy kolumny."
                    
                if error:
                    st.error(error)
                    return
                elif not urls:
                    st.warning("Nie znaleziono żadnych URL-i zdjęć.")
                    return
                
                # Zapisujemy początkowy stan
                save_processing_state(
                    session_id,
                    urls,
                    [],
                    {},
                    st.session_state.file_info,
                    st.session_state.processing_params
                )
                
                # Rozpocznij przetwarzanie
                with tempfile.TemporaryDirectory() as tmpdirname:
                    status_text.text(f"Inicjalizacja przetwarzania {len(urls)} obrazów...")
                    queue_info.text("")
                    start_time = time.time()
                    st.session_state["last_logs"] = []
                    st.session_state["output_bytes"] = None
                    st.session_state["output_filetype"] = file_type
                    st.session_state["output_filename"] = st.session_state.file_info["name"]
                    st.session_state["output_log"] = None
                    st.session_state["output_new_urls_map"] = None

                    # ---
                    def progress_callback(progress, processed, uploaded, total):
                        percent = int(progress*100)
                        elapsed = time.time() - start_time
                        speed = uploaded/elapsed if elapsed > 0 else 0
                        remaining = (total-processed)/speed if speed > 0 else 0
                        status_text.text(f"Przetwarzanie... {processed}/{total} ({percent}%) | Przesłano na FTP: {uploaded} | Szybkość: {speed:.1f} obrazów/s | ~{int(remaining/60)}m {int(remaining%60)}s pozostało")
                        progress_bar.progress(progress)
                    # ---
                    def queue_callback(info):
                        queue_info.text(info)
                    # ---
                    # Ulepszona funkcja przetwarzania równoległego
                    new_urls_map, downloaded_images, failed_urls = process_images_in_parallel(
                        urls,
                        tmpdirname,
                        st.session_state.ftp_settings,
                        max_workers=max_workers,
                        debug_container=debug_area,
                        max_retries=3,
                        progress_callback=progress_callback,
                        session_id=session_id
                    )
                    # ---
                    progress_bar.progress(1.0)
                    status_text.text(f"Zakończono przetwarzanie. Pobrano i przesłano {len(new_urls_map)} z {len(urls)} obrazów.")
                    
                    # Zapisz stan
                    processed_urls = list(new_urls_map.keys())
                    remaining_urls = [url for url in urls if url not in processed_urls]
                    save_processing_state(
                        session_id, urls, processed_urls, new_urls_map, 
                        st.session_state.file_info, st.session_state.processing_params
                    )
                    
                    # Raportuj błędy
                    if failed_urls:
                        with st.expander(f"Nie udało się przetworzyć {len(failed_urls)} URL-i"):
                            for fail in failed_urls[:20]:
                                st.warning(f"{fail['url']}: {fail['error']}")
                            if len(failed_urls) > 20:
                                st.warning(f"... oraz {len(failed_urls) - 20} więcej.")
                    
                    # Aktualizacja pliku
                    if new_urls_map:
                        # Wybór odpowiedniej funkcji aktualizacji
                        if file_type == "xml":
                            updated_content, error = update_xml_with_new_urls(
                                file_content, xpath, new_urls_map, new_node_name, separator
                            )
                        else:
                            updated_content, error = update_csv_with_new_urls(
                                file_content, column_name, new_urls_map, new_column_name, separator
                            )

                        if error:
                            st.error(f"Błąd podczas aktualizacji pliku: {error}")
                        else:
                            # Kodowanie wyniku
                            st.session_state.output_bytes = updated_content.encode(
                                st.session_state.file_info["encoding"]
                            )
                            st.success("Plik został zaktualizowany o nowe linki FTP.")
                            
                            # Google Drive
                            try:
                                success, message = save_to_google_drive(
                                    st.session_state.output_bytes,
                                    st.session_state.file_info,
                                    new_urls_map
                                )
                                
                                if success:
                                    st.success(f"✅ {message}")
                                else:
                                    st.warning(f"⚠️ {message}")
                                
                            except Exception as e:
                                st.error(f"Błąd Google Drive: {str(e)}")

                            # Przycisk pobierania
                            original_name = st.session_state.file_info["name"]
                            base_name = os.path.splitext(original_name)[0]
                            st.download_button(
                                label="📁 Pobierz zaktualizowany plik",
                                data=st.session_state.output_bytes,
                                file_name=f"{base_name}_updated.{file_type}",
                                mime="text/plain"
                            )
                            
                            # Jeśli są nieudane URL-e, zaproponuj ponowienie
                            if failed_urls and len(failed_urls) > 0:
                                st.warning(f"Nie udało się przetworzyć {len(failed_urls)} obrazów. Możesz wznowić przetwarzanie z zakładki 'Wznów przetwarzanie'.")

                    if st.button("Rozpocznij nową operację"):
                        reset_app_state()

            except (KeyboardInterrupt, SystemExit, st.runtime.scriptrunner.StopException):
                st.warning("Przetwarzanie zostało przerwane. Możesz je wznowić później z zakładki 'Wznów przetwarzanie'.")
                return
            except Exception as e:
                st.error(f"Wystąpił błąd: {str(e)}")
                st.info("Możesz spróbować wznowić przetwarzanie z zakładki 'Wznów przetwarzanie'.")
                return

    with tab2:
        st.subheader("Wznów wcześniej przerwane przetwarzanie")
        
        saved_sessions = list_saved_sessions()
        if not saved_sessions:
            st.info("Nie znaleziono zapisanych sesji przetwarzania.")
        else:
            st.write("Wybierz sesję do wznowienia:")
            
            # Tabela sesji
            sessions_df = pd.DataFrame(saved_sessions)
            if not sessions_df.empty:
                st.dataframe(sessions_df[["timestamp", "file_info", "progress", "percentage"]])
                
                # Wybór sesji
                selected_session_id = st.selectbox(
                    "Wybierz ID sesji do wznowienia:", 
                    options=[s["session_id"] for s in saved_sessions],
                    format_func=lambda x: f"{next((s['timestamp'] for s in saved_sessions if s['session_id'] == x), '')} - {next((s['file_info'] for s in saved_sessions if s['session_id'] == x), '')}"
                )
                
                max_workers_resume = st.slider(
                    "Liczba równoległych procesów", 
                    min_value=FTP_CONFIG['MIN_WORKERS'],
                    max_value=FTP_CONFIG['MAX_WORKERS'],
                    value=FTP_CONFIG['DEFAULT_WORKERS']
                )
                
                if st.button("Wznów przetwarzanie"):
                    state = load_processing_state(selected_session_id)
                    if state:
                        with tempfile.TemporaryDirectory() as tmpdirname:
                            resume_processing(state, tmpdirname, st.session_state.ftp_settings, max_workers=max_workers_resume)
                    else:
                        st.error("Nie udało się załadować stanu sesji.")

    with tab3:
        st.markdown("""
        ### Jak korzystać z aplikacji

        1. **Wgraj plik XML lub CSV** - aplikacja automatycznie wykryje kodowanie
        2. **Skonfiguruj pobieranie zdjęć**:
           - Dla XML: Podaj XPath do węzła zawierającego URL-e zdjęć
           - Dla CSV: Podaj nazwę kolumny zawierającej URL-e zdjęć
           - Określ separator, jeśli w jednej komórce/węźle znajduje się wiele URL-i
        3. **Skonfiguruj serwer FTP** - podaj dane dostępowe do serwera FTP
           - Podaj ścieżkę HTTP, pod którą będą dostępne zdjęcia (np. https://example.com/images/)
        4. **Pobierz zdjęcia i prześlij na FTP** - aplikacja pobierze zdjęcia i prześle je na serwer FTP
           - Możesz dostosować liczbę równoległych procesów pobierania dla przyspieszenia procesu
        5. **Pobierz zaktualizowany plik** - plik źródłowy zostanie zaktualizowany o nowe linki HTTP/FTP
        6. **Wznawianie przerwanego procesu**:
           - W przypadku błędu lub przerwania procesu, przejdź do zakładki "Wznów przetwarzanie"
           - Wybierz odpowiednią sesję i kontynuuj pobieranie od miejsca przerwania

        ### Przykłady konfiguracji

        #### XML

        - XPath: //product/image
        - Nazwa nowego węzła: ftp
        - Ścieżka HTTP: https://example.com/images/

        #### CSV

        - Nazwa kolumny: image_url
        - Nazwa nowej kolumny: ftp_image_url
        - Ścieżka HTTP: https://example.com/images/

        ### Obsługiwane formaty

        - **XML** - pliki XML z linkami do zdjęć w określonych węzłach
        - **CSV** - pliki CSV z linkami do zdjęć w określonych kolumnach
        """)

if __name__ == "__main__":
    main()
