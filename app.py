import streamlit as st
import requests
import tempfile
import os
import re
import traceback
import json
from datetime import datetime
import pandas as pd
import io
import ftplib
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import shutil
import uuid

# --- Funkcje pomocnicze ---

def authenticate_user():
    """Uwierzytelnianie użytkownika"""
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
    """Inicjalizacja zmiennych sesji"""
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = ""
    if "edited_code" not in st.session_state:
        st.session_state.edited_code = ""
    if "output_bytes" not in st.session_state:
        st.session_state.output_bytes = None
    if "file_info" not in st.session_state:
        st.session_state.file_info = None
    if "show_editor" not in st.session_state:
        st.session_state.show_editor = False
    if "error_info" not in st.session_state:
        st.session_state.error_info = None
    if "code_fixed" not in st.session_state:
        st.session_state.code_fixed = False
    if "fix_requested" not in st.session_state:
        st.session_state.fix_requested = False
    if "downloaded_images" not in st.session_state:
        st.session_state.downloaded_images = []
    if "ftp_settings" not in st.session_state:
        st.session_state.ftp_settings = {
            "host": "",
            "port": 21,
            "username": "",
            "password": "",
            "directory": "/"
        }


def read_file_content(uploaded_file):
    """Czyta zawartość pliku z obsługą różnych kodowań"""
    if not uploaded_file:
        return None, "Nie wybrano pliku"

    try:
        raw_bytes = uploaded_file.read()
        file_type = uploaded_file.name.split(".")[-1].lower()

        if file_type not in ["xml", "csv"]:
            return None, "Nieobsługiwany typ pliku. Akceptowane formaty to XML i CSV."

        # Autodetekcja kodowania dla XML
        if file_type == "xml":
            encoding_declared = re.search(br'<\?xml[^>]*encoding=["\']([^"\']+)["\']', raw_bytes)
            encodings_to_try = [encoding_declared.group(1).decode('ascii')] if encoding_declared else []
        else:
            encodings_to_try = []

        # Lista kodowań do próbowania
        encodings_to_try += ["utf-8", "iso-8859-2", "windows-1250", "utf-16"]

        for enc in encodings_to_try:
            try:
                file_contents = raw_bytes.decode(enc)
                return {"content": file_contents, "raw_bytes": raw_bytes, 
                        "type": file_type, "encoding": enc, "name": uploaded_file.name}, None
            except UnicodeDecodeError:
                continue

        # Jeśli żadne kodowanie nie działa, spróbuj wczytać jako binarny
        if file_type == "csv":
            try:
                buffer = io.BytesIO(raw_bytes)
                df = pd.read_csv(buffer, sep=None, engine='python')
                file_contents = df.to_csv(index=False)
                return {"content": file_contents, "raw_bytes": raw_bytes, 
                        "type": file_type, "encoding": "auto-detected", 
                        "name": uploaded_file.name, "dataframe": df}, None
            except Exception as e:
                pass

        return None, "Nie udało się odczytać pliku – nieznane kodowanie."

    except Exception as e:
        return None, f"Błąd podczas odczytu pliku: {str(e)}"


def download_image(url, temp_dir):
    """Pobiera obraz z URL i zapisuje go w katalogu tymczasowym"""
    try:
        # Sprawdź czy URL jest poprawny
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return None, f"Nieprawidłowy URL: {url}"

        # Pobierz obraz
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Określ nazwę pliku
        if "Content-Disposition" in response.headers:
            content_disp = response.headers["Content-Disposition"]
            filename_match = re.search(r'filename="?([^"]+)"?', content_disp)
            if filename_match:
                filename = filename_match.group(1)
            else:
                filename = os.path.basename(parsed_url.path) or f"image_{uuid.uuid4().hex}"
        else:
            filename = os.path.basename(parsed_url.path) or f"image_{uuid.uuid4().hex}"

        # Dodaj rozszerzenie, jeśli go nie ma
        if not os.path.splitext(filename)[1]:
            content_type = response.headers.get("Content-Type", "")
            if "jpeg" in content_type or "jpg" in content_type:
                filename += ".jpg"
            elif "png" in content_type:
                filename += ".png"
            elif "gif" in content_type:
                filename += ".gif"
            elif "webp" in content_type:
                filename += ".webp"
            else:
                filename += ".jpg"  # Domyślne rozszerzenie

        # Zapisz plik
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)

        return {"path": file_path, "filename": filename, "original_url": url}, None

    except requests.exceptions.RequestException as e:
        return None, f"Błąd podczas pobierania obrazu {url}: {str(e)}"
    except Exception as e:
        return None, f"Nieoczekiwany błąd: {str(e)}"


def upload_to_ftp(file_path, ftp_settings, remote_filename=None):
    """Przesyła plik na serwer FTP"""
    try:
        # Połącz z serwerem FTP
        ftp = ftplib.FTP()
        ftp.connect(ftp_settings["host"], ftp_settings["port"])
        ftp.login(ftp_settings["username"], ftp_settings["password"])

        # Przejdź do katalogu docelowego
        if ftp_settings["directory"] and ftp_settings["directory"] != "/":
            try:
                ftp.cwd(ftp_settings["directory"])
            except ftplib.error_perm:
                # Jeśli katalog nie istnieje, spróbuj go utworzyć
                dirs = ftp_settings["directory"].strip("/").split("/")
                for directory in dirs:
                    if directory:
                        try:
                            ftp.cwd(directory)
                        except ftplib.error_perm:
                            ftp.mkd(directory)
                            ftp.cwd(directory)

        # Określ nazwę pliku na serwerze
        if not remote_filename:
            remote_filename = os.path.basename(file_path)

        # Przesyłanie pliku
        with open(file_path, 'rb') as file:
            ftp.storbinary(f'STOR {remote_filename}', file)

        # Pobierz URL do pliku
        file_url = f"ftp://{ftp_settings['username']}:***@{ftp_settings['host']}"
        if ftp_settings["directory"] and ftp_settings["directory"] != "/":
            file_url += f"{ftp_settings['directory']}"
        if not file_url.endswith("/"):
            file_url += "/"
        file_url += remote_filename

        # Zamknij połączenie
        ftp.quit()

        return {"success": True, "url": file_url, "filename": remote_filename}

    except Exception as e:
        return {"success": False, "error": str(e)}


def extract_image_urls_from_xml(xml_content, xpath, separator=","):
    """Wyciąga URL-e obrazów z pliku XML"""
    try:
        root = ET.fromstring(xml_content)
        urls = []

        # Znajdź wszystkie elementy pasujące do XPath
        elements = root.findall(xpath)

        for element in elements:
            element_text = element.text
            if element_text:
                # Podziel tekst, jeśli zawiera separator
                if separator in element_text:
                    for url in element_text.split(separator):
                        url = url.strip()
                        if url:
                            urls.append(url)
                else:
                    urls.append(element_text.strip())

        return urls, None

    except Exception as e:
        return None, f"Błąd podczas parsowania XML: {str(e)}"


def extract_image_urls_from_csv(csv_content, column_name, separator=","):
    """Wyciąga URL-e obrazów z pliku CSV"""
    try:
        df = pd.read_csv(io.StringIO(csv_content))

        if column_name not in df.columns:
            return None, f"Kolumna '{column_name}' nie istnieje w pliku CSV."

        urls = []

        for value in df[column_name]:
            if pd.notna(value):
                # Podziel tekst, jeśli zawiera separator
                if separator in str(value):
                    for url in str(value).split(separator):
                        url = url.strip()
                        if url:
                            urls.append(url)
                else:
                    urls.append(str(value).strip())

        return urls, None

    except Exception as e:
        return None, f"Błąd podczas parsowania CSV: {str(e)}"


def update_xml_with_new_urls(xml_content, xpath, new_urls_map, new_node_name):
    """Aktualizuje plik XML, dodając nowe węzły z URL-ami FTP"""
    try:
        root = ET.fromstring(xml_content)

        # Znajdź wszystkie elementy pasujące do XPath
        elements = root.findall(xpath)

        for element in elements:
            if element.text in new_urls_map:
                # Znajdź rodzica elementu
                parent = None
                for potential_parent in root.iter():
                    if element in list(potential_parent):
                        parent = potential_parent
                        break

                if parent is not None:
                    # Utwórz nowy element z tym samym tagiem lub określonym tagiem
                    new_element = ET.Element(new_node_name)
                    new_element.text = new_urls_map[element.text]

                    # Dodaj nowy element do tego samego rodzica
                    parent_index = list(parent).index(element)
                    parent.insert(parent_index + 1, new_element)

        # Konwertuj zaktualizowany XML na string
        return ET.tostring(root, encoding='unicode'), None

    except Exception as e:
        return None, f"Błąd podczas aktualizacji XML: {str(e)}"


def update_csv_with_new_urls(csv_content, column_name, new_urls_map, new_column_name):
    """Aktualizuje plik CSV, dodając nową kolumnę z URL-ami FTP"""
    try:
        df = pd.read_csv(io.StringIO(csv_content))

        if column_name not in df.columns:
            return None, f"Kolumna '{column_name}' nie istnieje w pliku CSV."

        # Utwórz nową kolumnę
        df[new_column_name] = ""

        # Zaktualizuj wartości w nowej kolumnie
        for idx, value in enumerate(df[column_name]):
            if pd.notna(value) and value in new_urls_map:
                df.at[idx, new_column_name] = new_urls_map[value]

        # Konwertuj zaktualizowany DataFrame na string
        return df.to_csv(index=False), None

    except Exception as e:
        return None, f"Błąd podczas aktualizacji CSV: {str(e)}"


def reset_app_state():
    """Resetuje stan aplikacji"""
    for key in list(st.session_state.keys()):
        if key not in ["authenticated", "ftp_settings"]:
            del st.session_state[key]
    initialize_session_state()
    st.rerun()


def main():
    """Główna funkcja aplikacji"""
    # Ustawienia strony
    st.set_page_config(page_title="Pobieranie zdjęć z XML/CSV", layout="centered")
    # Uwierzytelnianie
    authenticate_user()

    # Inicjalizacja stanu sesji
    initialize_session_state()

    # Interfejs użytkownika - prosty layout
    st.title("Pobieranie zdjęć z XML/CSV")

    # Zakładki
    tab1, tab2 = st.tabs(["Pobieranie zdjęć", "Pomoc"])

    with tab1:
        st.markdown("""
        To narzędzie umożliwia pobieranie zdjęć z plików XML lub CSV i zapisywanie ich na serwerze FTP.
        Prześlij plik, wskaż lokalizację linków do zdjęć, podaj dane FTP i pobierz zdjęcia.
        """)

        # Sekcja 1: Wczytywanie pliku
        st.subheader("1. Wczytaj plik źródłowy")
        uploaded_file = st.file_uploader("Wgraj plik XML lub CSV", type=["xml", "csv"])

        if uploaded_file:
            file_info, error = read_file_content(uploaded_file)
            if error:
                st.error(error)
            else:
                st.success(f"Wczytano plik: {file_info['name']} ({file_info['type'].upper()}, {file_info['encoding']})")
                st.session_state.file_info = file_info

        # Sekcja 2: Konfiguracja pobierania
        st.subheader("2. Konfiguracja pobierania zdjęć")

        if st.session_state.file_info:
            file_type = st.session_state.file_info["type"]

            if file_type == "xml":
                xpath = st.text_input(
                    "XPath do węzła zawierającego URL-e zdjęć", 
                    placeholder="Np. //product/image"
                )
                new_node_name = st.text_input(
                    "Nazwa nowego węzła dla linków FTP", 
                    placeholder="Np. ftp_image"
                )
            else:  # csv
                column_name = st.text_input(
                    "Nazwa kolumny zawierającej URL-e zdjęć", 
                    placeholder="Np. image_url"
                )
                new_column_name = st.text_input(
                    "Nazwa nowej kolumny dla linków FTP", 
                    placeholder="Np. ftp_image_url"
                )

            separator = st.text_input(
                "Separator URL-i (jeśli w jednej komórce/węźle jest wiele linków)", 
                value=","
            )

        # Sekcja 3: Konfiguracja FTP
        st.subheader("3. Konfiguracja serwera FTP")

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.ftp_settings["host"] = st.text_input(
                "Adres serwera FTP", 
                value=st.session_state.ftp_settings["host"]
            )
            st.session_state.ftp_settings["port"] = st.number_input(
                "Port", 
                value=st.session_state.ftp_settings["port"],
                min_value=1, 
                max_value=65535
            )
            st.session_state.ftp_settings["directory"] = st.text_input(
                "Katalog docelowy", 
                value=st.session_state.ftp_settings["directory"]
            )

        with col2:
            st.session_state.ftp_settings["username"] = st.text_input(
                "Nazwa użytkownika", 
                value=st.session_state.ftp_settings["username"]
            )
            st.session_state.ftp_settings["password"] = st.text_input(
                "Hasło", 
                type="password", 
                value=st.session_state.ftp_settings["password"]
            )

        # Sekcja 4: Pobieranie i przesyłanie
        st.subheader("4. Pobierz zdjęcia i prześlij na FTP")

        if st.session_state.file_info and st.button("Pobierz zdjęcia i prześlij na FTP"):
            file_type = st.session_state.file_info["type"]
            file_content = st.session_state.file_info["content"]

            # Pobierz URL-e zdjęć
            if file_type == "xml" and xpath:
                urls, error = extract_image_urls_from_xml(file_content, xpath, separator)
            elif file_type == "csv" and column_name:
                urls, error = extract_image_urls_from_csv(file_content, column_name, separator)
            else:
                urls, error = None, "Nie podano ścieżki XPath lub nazwy kolumny."

            if error:
                st.error(error)
            elif not urls:
                st.warning("Nie znaleziono żadnych URL-i zdjęć.")
            else:
                # Sprawdź konfigurację FTP
                if not st.session_state.ftp_settings["host"] or not st.session_state.ftp_settings["username"]:
                    st.error("Podaj dane serwera FTP.")
                else:
                    # Pobierz i prześlij zdjęcia
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    with tempfile.TemporaryDirectory() as tmpdirname:
                        downloaded_images = []
                        new_urls_map = {}  # Mapowanie oryginalny URL -> nowy URL FTP

                        for i, url in enumerate(urls):
                            status_text.text(f"Przetwarzanie {i+1}/{len(urls)}: {url}")
                            progress_bar.progress((i) / len(urls))

                            # Pobierz obraz
                            image_info, error = download_image(url, tmpdirname)
                            if error:
                                st.warning(f"Nie udało się pobrać {url}: {error}")
                                continue

                            # Prześlij na FTP
                            upload_result = upload_to_ftp(
                                image_info["path"], 
                                st.session_state.ftp_settings
                            )

                            if upload_result["success"]:
                                # Dodaj do listy pobranych i przesłanych obrazów
                                downloaded_images.append({
                                    "original_url": url,
                                    "ftp_url": upload_result["url"],
                                    "filename": upload_result["filename"]
                                })

                                # Dodaj do mapy URL-i
                                ftp_url = f"ftp://{st.session_state.ftp_settings['host']}"
                                if st.session_state.ftp_settings["directory"] and st.session_state.ftp_settings["directory"] != "/":
                                    ftp_url += f"{st.session_state.ftp_settings['directory']}"
                                if not ftp_url.endswith("/"):
                                    ftp_url += "/"
                                ftp_url += upload_result["filename"]
                                new_urls_map[url] = ftp_url
                            else:
                                st.warning(f"Nie udało się przesłać {url} na FTP: {upload_result['error']}")

                        progress_bar.progress(1.0)
                        status_text.text(f"Zakończono przetwarzanie. Pobrano i przesłano {len(downloaded_images)} z {len(urls)} zdjęć.")

                        # Zapisz informacje o pobranych obrazach
                        st.session_state.downloaded_images = downloaded_images

                        # Aktualizuj plik źródłowy o nowe URL-e
                        if new_urls_map:
                            if file_type == "xml":
                                updated_content, error = update_xml_with_new_urls(
                                    file_content, 
                                    xpath, 
                                    new_urls_map, 
                                    new_node_name
                                )
                            else:  # csv
                                updated_content, error = update_csv_with_new_urls(
                                    file_content, 
                                    column_name, 
                                    new_urls_map, 
                                    new_column_name
                                )

                            if error:
                                st.error(f"Błąd podczas aktualizacji pliku: {error}")
                            else:
                                # Zapisz zaktualizowany plik
                                st.session_state.output_bytes = updated_content.encode(
                                    st.session_state.file_info["encoding"]
                                )
                                st.success("Plik został zaktualizowany o nowe linki FTP.")

                                # Przycisk pobierania
                                original_name = st.session_state.file_info["name"]
                                base_name = os.path.splitext(original_name)[0]

                                st.download_button(
                                    label=f"📁 Pobierz zaktualizowany plik",
                                    data=st.session_state.output_bytes,
                                    file_name=f"{base_name}_updated.{file_type}",
                                    mime="text/plain"
                                )

        # Wyświetl listę pobranych zdjęć
        if st.session_state.downloaded_images:
            st.subheader("Pobrane i przesłane zdjęcia")

            for i, image in enumerate(st.session_state.downloaded_images):
                st.markdown(f"**{i+1}. {image['filename']}**")
                st.markdown(f"- Oryginalny URL: {image['original_url']}")
                st.markdown(f"- FTP URL: {image['ftp_url']}")

            # Przycisk do resetowania stanu aplikacji
            if st.button("Rozpocznij nową operację"):
                reset_app_state()

    with tab2:
        st.markdown("""
        ### Jak korzystać z aplikacji

        1. **Wgraj plik XML lub CSV** - aplikacja automatycznie wykryje kodowanie
        2. **Skonfiguruj pobieranie zdjęć**:
           - Dla XML: Podaj XPath do węzła zawierającego URL-e zdjęć
           - Dla CSV: Podaj nazwę kolumny zawierającej URL-e zdjęć
           - Określ separator, jeśli w jednej komórce/węźle znajduje się wiele URL-i
        3. **Skonfiguruj serwer FTP** - podaj dane dostępowe do serwera FTP
        4. **Pobierz zdjęcia i prześlij na FTP** - aplikacja pobierze zdjęcia i prześle je na serwer FTP
        5. **Pobierz zaktualizowany plik** - plik źródłowy zostanie zaktualizowany o nowe linki FTP

        ### Przykłady konfiguracji

        #### XML

        - XPath: `//product/image`
        - Nazwa nowego węzła: `ftp_image`

        #### CSV

        - Nazwa kolumny: `image_url`
        - Nazwa nowej kolumny: `ftp_image_url`

        ### Obsługiwane formaty

        - **XML** - pliki XML z linkami do zdjęć w określonych węzłach
        - **CSV** - pliki CSV z linkami do zdjęć w określonych kolumnach
        """)

if __name__ == "__main__":
    main()