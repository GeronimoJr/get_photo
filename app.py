import streamlit as st
import requests, tempfile, os, re, json, time, io, ftplib, logging, uuid, codecs, concurrent.futures
from datetime import datetime
from urllib.parse import urlparse
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s"
)

class FTPManager:
    def __init__(self, s):
        self.s=s; self.ftp=None; self.c=False
    def connect(self):
        if self.c: return True
        try:
            self.ftp=ftplib.FTP(); self.ftp.connect(self.s["host"], self.s["port"]); self.ftp.login(self.s["username"], self.s["password"])
            d=self.s["directory"].strip("/")
            if d:
                for p in d.split("/"):
                    try: self.ftp.cwd(p)
                    except ftplib.error_perm: self.ftp.mkd(p); self.ftp.cwd(p)
            self.c=True; return True
        except Exception as e:
            logging.error(f"FTP connect error: {e}"); self.c=False; return False
    def upload_file(self, fp, rn=None):
        if not self.c and not self.connect(): return {"success":False,"error":"FTP connect failed"}
        if not os.path.exists(fp) or os.path.getsize(fp)==0: return {"success":False,"error":f"Empty or missing file: {fp}"}
        try:
            if not rn: rn=os.path.basename(fp)
            with open(fp,"rb") as f: self.ftp.storbinary(f"STOR {rn}", f)
            hp=self.s.get("http_path","").rstrip("/")+"/" if self.s.get("http_path") else ""
            if hp:
                url=hp+rn
            else:
                url=f"ftp://{self.s['host']}/{self.s['directory'].strip('/')}/{rn}"
            return {"success":True,"url":url,"filename":rn}
        except Exception as e:
            logging.exception("FTP upload")
            self.c=False
            if not self.connect(): return {"success":False,"error":f"Lost connection: {e}"}
            try:
                with open(fp,"rb") as f: self.ftp.storbinary(f"STOR {rn}", f)
                return {"success":True,"url":url,"filename":rn}
            except Exception as e2:
                logging.exception("FTP reupload"); return {"success":False,"error":"Reupload failed"}

    def close(self):
        if self.c:
            try: self.ftp.quit()
            except: pass
            self.c=False

def authenticate_user():
    if "auth" not in st.session_state: st.session_state.auth=False
    if not st.session_state.auth:
        st.title("Logowanie"); u=st.text_input("Login"); p=st.text_input("Hasło",type="password")
        if st.button("Zaloguj"):
            if u==st.secrets.APP_USER and p==st.secrets.APP_PASSWORD:
                st.session_state.auth=True; st.rerun()
            else: st.error("Błędne dane")
        st.stop()

def init_state():
    d={"generated_code":"","edited_code":"","output_bytes":None,"file_info":None,"ftp_settings":{"host":"","port":21,"username":"","password":"","directory":"/","http_path":""},"processing_params":{"xpath":"","column_name":"","new_node_name":"","new_column_name":"","separator":","}}
    for k,v in d.items():
        if k not in st.session_state: st.session_state[k]=v

def save_state(sid, all_urls, processed, new_map, file_info, params, failed=None):
    if failed is None: failed=[]
    rem=[u for u in all_urls if u not in processed]
    state={"session_id":sid,"timestamp":datetime.now().isoformat(),"file_info":{"name":file_info["name"],"type":file_info["type"],"encoding":file_info["encoding"]},"processing_params":params,"all_urls":all_urls,"processed_urls":processed,"failed_urls":failed,"remaining_urls":rem,"new_urls_map":new_map,"total_urls":len(all_urls),"file_content":file_info.get("content","")}
    d=os.path.join(os.path.expanduser("~"),".xml_image_processor"); os.makedirs(d,exist_ok=True)
    with open(os.path.join(d,f"session_{sid}.json"),"w") as f: json.dump(state,f)

def load_state(sid=None):
    d=os.path.join(os.path.expanduser("~"),".xml_image_processor")
    if not os.path.exists(d): return None
    if sid:
        p=os.path.join(d,f"session_{sid}.json")
        if os.path.exists(p): return json.load(open(p))
        return None
    fs=[f for f in os.listdir(d) if f.startswith("session_")]
    if not fs: return None
    fs.sort(key=lambda x:os.path.getmtime(os.path.join(d,x)),reverse=True)
    return json.load(open(os.path.join(d,fs[0])))

def list_sessions():
    d=os.path.join(os.path.expanduser("~"),".xml_image_processor")
    out=[]
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.startswith("session_"):
                try:
                    stt=json.load(open(os.path.join(d,f)))
                    pct=round(len(stt["processed_urls"])/stt["total_urls"]*100,1)
                    out.append({"session_id":stt["session_id"],"timestamp":stt["timestamp"],"file":stt["file_info"]["name"],"progress":f"{len(stt['processed_urls'])}/{stt['total_urls']}","percentage":pct})
                except: pass
    out.sort(key=lambda x:x["timestamp"],reverse=True)
    return out

def download_image(url,td):
    try:
        p=urlparse(url)
        hdr={"User-Agent":"Mozilla/5.0","Accept":"*/*","Referer":f"{p.scheme}://{p.netloc}/"}
        if "image_show.php" in url:
            r=requests.get(url,headers=hdr,timeout=10); r.raise_for_status()
            s=BeautifulSoup(r.text,"html.parser"); img=s.find("img")
            if not img or not img.get("src"): return None,"No <img>"
            src=img["src"]; img_url=src if src.startswith("http") else f"{p.scheme}://{p.netloc}/{src.lstrip('/')}"
        else: img_url=url
        for _ in range(3):
            try:
                r2=requests.get(img_url,headers=hdr,timeout=15); r2.raise_for_status()
                ct=r2.headers.get("Content-Type","")
                if not ct.startswith("image/"): continue
                ext={ "image/jpeg":".jpg","image/png":".png","image/gif":".gif","image/webp":".webp"}.get(ct,".jpg")
                fn=f"img_{uuid.uuid4().hex}{ext}"; fp=os.path.join(td,fn)
                with open(fp,"wb") as f: f.write(r2.content)
                if os.path.getsize(fp)>100: return {"path":fp,"filename":fn},None
                return None,"Empty file"
            except Exception as e:
                if _==2: return None,str(e)
        return None,"Download error"
    except Exception as e:
        logging.exception("download_image"); return None,str(e)

def process_parallel(urls,td,ftp_s,max_workers,debug=None):
    nm,dl,fail={},[],[]
    def proc(u):
        img,err=download_image(u,td)
        if err: return {"status":"download_error","url":u,"error":err}
        fm=FTPManager(ftp_s)
        if not fm.connect(): return {"status":"ftp_error","url":u,"error":"FTP connect"}
        res=fm.upload_file(img["path"]); fm.close(); time.sleep(0.5)
        if res["success"]: return {"status":"success","url":u,"ftp_url":res["url"],"filename":res["filename"]}
        return {"status":"upload_error","url":u,"error":res.get("error","")}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut={ex.submit(proc,u):u for u in urls}
        for f in concurrent.futures.as_completed(fut):
            r=f.result()
            if r["status"]=="success":
                nm[r["url"]]=r["ftp_url"]; dl.append({"original_url":r["url"],"ftp_url":r["ftp_url"],"filename":r["filename"]})
                if debug: debug.success(f"Pobrano: {r['url']}")
            else:
                fail.append({"url":r["url"],"error":r.get("error","")})
                if debug: debug.warning(f"Błąd {r['url']}: {r.get('error')}")
    return nm, dl, fail

def extract_xml(c,xp,sep):
    try:
        c=c.lstrip("\ufeff"); c=re.sub(r'[\x00-\x1F\x7F]','',c)
        if xp in ["//product/image","product/image"]:
            m=re.findall(r'<image>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</image>',c,re.DOTALL)
            return [u.replace('&amp;','&').strip() for u in m if ('http://' in u or 'https://' in u)],None
        rt=ET.fromstring(c); pb=xp[2:] if xp.startswith("//") else xp
        els=rt.findall(f".//{pb}"); out=[]
        for el in els:
            t=el.attrib.get(xp.split("/@")[-1],"") if "/@" in xp else el.text or ""
            for u in (t.split(sep) if sep in t else [t]):
                u=u.replace("&amp;","&").strip()
                if "http://" in u or "https://" in u: out.append(u)
        return out,None
    except Exception as e:
        logging.exception("extract_xml"); return None,str(e)

def update_xml(c,xp,map_,nn,sep):
    try:
        rt=ET.fromstring(c); pb=xp[2:] if xp.startswith("//") else xp
        els=rt.findall(f".//{pb}"); pm={c:p for p in rt.iter() for c in p}
        for el in els:
            orig=el.attrib.get(xp.split("/@")[-1],"").strip() if "/@" in xp else (el.text or "").strip()
            if not orig: continue
            us=orig.split(sep) if sep in orig else [orig]
            parent=pm.get(el); ftp_p=parent.find("ftp_images") or parent.append(ET.Element("ftp_images"))
            for u in us:
                if u in map_:
                    nn_el=ET.Element(nn); nn_el.text=map_[u]; ftp_p.append(nn_el)
        return ET.tostring(rt,encoding="unicode"),None
    except Exception as e:
        logging.exception("update_xml"); return None,str(e)

def extract_csv(c,cn,sep):
    try:
        df=pd.read_csv(io.StringIO(c)); out=[]
        for v in df[cn]:
            for u in (str(v).split(sep) if sep in str(v) else [v]):
                u=str(u).strip(); 
                if u: out.append(u)
        return out,None
    except Exception as e:
        logging.exception("extract_csv"); return None,str(e)

def update_csv(c,cn,map_,nn,sep):
    try:
        df=pd.read_csv(io.StringIO(c))
        if nn not in df.columns: df[nn]=""
        for i,v in enumerate(df[cn]):
            for u in (str(v).split(sep) if sep in str(v) else [v]):
                if u in map_: df.at[i,nn]=map_[u]
        return df.to_csv(index=False),None
    except Exception as e:
        logging.exception("update_csv"); return None,str(e)

def save_to_drive(b,f,map_):
    try:
        fid=st.secrets.GOOGLE_DRIVE_FOLDER_ID; cred=json.loads(st.secrets.GOOGLE_DRIVE_CREDENTIALS_JSON)
        creds=ServiceAccountCredentials.from_json_keyfile_dict(cred,["https://www.googleapis.com/auth/drive"])
        ga=GoogleAuth(); ga.credentials=creds; dr=GoogleDrive(ga)
        now=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tmpd=tempfile.gettempdir()
        rp=os.path.join(tmpd,f"out.{f['type']}"); lp=os.path.join(tmpd,"log.txt")
        with open(rp,"wb") as fo: fo.write(b)
        with open(lp,"w",encoding="utf-8") as fo: 
            for o,n in map_.items(): fo.write(f"{o} -> {n}\n")
        lf=dr.CreateFile({"title":f"log_{now}.txt","parents":[{"id":fid}],"mimeType":"text/plain"}); lf.SetContentFile(lp); lf.Upload()
        rf=dr.CreateFile({"title":f"res_{now}.{f['type']}","parents":[{"id":fid}],"mimeType":f"application/{f['type']}"}); rf.SetContentFile(rp); rf.Upload()
        return True,"Saved"
    except Exception as e:
        logging.exception("save_to_drive"); return False,str(e)

def main():
    st.set_page_config(page_title="Pobieranie zdjęć",layout="centered")
    authenticate_user(); init_state()
    tab1,tab2,tab3=st.tabs(["Pobieranie","Wznów","Pomoc"])
    with tab1:
        fu=st.file_uploader("Wgraj XML/CSV",type=["xml","csv"])
        if fu:
            raw=fu.read(); ft=fu.name.split(".")[-1].lower()
            fi=None; err=None
            if ft=="xml" or ft=="csv":
                for enc in ["utf-8","iso-8859-2","windows-1250","utf-16-le","utf-16-be"]:
                    try:
                        content=raw.decode(enc)
                        fi={"content":content,"raw_bytes":raw,"type":ft,"encoding":enc,"name":fu.name}
                        break
                    except: continue
                if not fi: err="Nie można odczytać pliku"
            else: err="Nieobsługiwany format"
            if err: st.error(err)
            else:
                st.success(f"Wczytano {fi['name']} ({fi['encoding']})"); st.session_state.file_info=fi
        if st.session_state.file_info:
            p=st.session_state.processing_params
            if st.session_state.file_info["type"]=="xml":
                p["xpath"]=st.text_input("XPath",value=p["xpath"])
                p["new_node_name"]=st.text_input("Nowy węzeł",value=p["new_node_name"])
            else:
                p["column_name"]=st.text_input("Kolumna",value=p["column_name"])
                p["new_column_name"]=st.text_input("Nowa kolumna",value=p["new_column_name"])
            p["separator"]=st.text_input("Separator",value=p["separator"])
            fs=st.session_state.ftp_settings
            c1,c2=st.columns(2)
            with c1:
                fs["host"]=st.text_input("FTP host",value=fs["host"])
                fs["port"]=st.number_input("Port",value=fs["port"],min_value=1,max_value=65535)
                fs["directory"]=st.text_input("Katalog",value=fs["directory"])
                fs["http_path"]=st.text_input("HTTP ścieżka",value=fs["http_path"])
            with c2:
                fs["username"]=st.text_input("FTP user",value=fs["username"])
                fs["password"]=st.text_input("FTP pass",type="password",value=fs["password"])
            max_w=st.slider("Równoległe",1,10,3)
            if st.button("Pobierz i prześlij"):
                fi=st.session_state.file_info; p=st.session_state.processing_params; fs=st.session_state.ftp_settings
                if (fi["type"]=="xml" and (not p["xpath"] or not p["new_node_name"])) or (fi["type"]=="csv" and (not p["column_name"] or not p["new_column_name"])):
                    st.error("Uzupełnij pola"); st.stop()
                if fi["type"]=="xml":
                    urls,err=extract_xml(fi["content"],p["xpath"],p["separator"])
                else:
                    urls,err=extract_csv(fi["content"],p["column_name"],p["separator"])
                if err: st.error(err); st.stop()
                if not urls: st.warning("Brak URL"); st.stop()
                st.success(f"{len(urls)} URL")
                with st.expander("Podgląd"):
                    for u in urls[:5]: st.write(u)
                    if len(urls)>5: st.write(f"...{len(urls)-5} więcej")
                if not fs["host"] or not fs["username"]: st.error("Brak FTP"); st.stop()
                prog=st.progress(0); status=st.empty(); dbg=st.empty()
                sid=f"{uuid.uuid4().hex}_{int(time.time())}"
                save_state(sid,urls,[],{},fi,p,[])
                all_dl=[]; all_fail=[]
                for i in range(0,len(urls),10):
                    batch=urls[i:i+10]
                    status.text(f"Paczka {i+1}-{min(i+10,len(urls))} z {len(urls)}")
                    prog.progress(i/len(urls))
                    nm,dl,fail=process_parallel(batch,tempfile.mkdtemp(),fs,max_w,dbg)
                    all_dl+=dl; all_fail+=fail
                    save_state(sid,urls,[u for u,_ in all_dl],{d["original_url"]:d["ftp_url"] for d in all_dl},fi,p,all_fail)
                prog.progress(1.0)
                status.text(f"Skończono: {len(all_dl)}/{len(urls)}")
                if all_dl:
                    if fi["type"]=="xml":
                        up,err=update_xml(fi["content"],p["xpath"],{d["original_url"]:d["ftp_url"] for d in all_dl},p["new_node_name"],p["separator"])
                    else:
                        up,err=update_csv(fi["content"],p["column_name"],{d["original_url"]:d["ftp_url"] for d in all_dl},p["new_column_name"],p["separator"])
                    if err: st.error(err)
                    else:
                        b=up.encode(fi["encoding"]); st.session_state.output_bytes=b
                        ok,msg=save_to_drive(b,fi,{d["original_url"]:d["ftp_url"] for d in all_dl})
                        st.success("Zapisano na Google Drive" if ok else f"Błąd Drive: {msg}")
                        st.download_button("Pobierz plik",data=b,file_name=f"{os.path.splitext(fi['name'])[0]}_upd.{fi['type']}")
    with tab2:
        st.subheader("Wznów")
        ss=list_sessions()
        if not ss: st.info("Brak sesji")
        else:
            df=pd.DataFrame(ss); st.dataframe(df[["timestamp","file","progress","percentage"]])
            sel=st.selectbox("Wybierz",options=[s["session_id"] for s in ss],format_func=lambda x: next(s["timestamp"] for s in ss if s["session_id"]==x))
            max_w2=st.slider("Równoległe",1,10,3,key="rw2")
            if st.button("Wznów"):
                stt=load_state(sel)
                if not stt: st.error("Brak stanu"); st.stop()
                urls=stt["remaining_urls"]; fi=stt["file_info"]; p=stt["processing_params"]; fs=st.session_state.ftp_settings
                prog=st.progress(len(stt["processed_urls"])/stt["total_urls"]); status=st.empty(); dbg=st.empty()
                all_dl=[{"original_url":u,"ftp_url":stt["new_urls_map"][u]} for u in stt["processed_urls"]]
                all_fail=stt["failed_urls"]
                for i in range(0,len(urls),10):
                    batch=urls[i:i+10]
                    status.text(f"Paczka {len(all_dl)+1}-{stt['total_urls']}")
                    prog.progress(len(all_dl)/stt["total_urls"])
                    nm,dl,fail=process_parallel(batch,tempfile.mkdtemp(),fs,max_w2,dbg)
                    all_dl+=dl; all_fail+=fail
                    save_state(sel,stt["all_urls"],[u["original_url"] for u in all_dl],{d["original_url"]:d["ftp_url"] for d in all_dl},fi,p,all_fail)
                prog.progress(1.0)
                status.text(f"Skończono: {len(all_dl)}/{stt['total_urls']}")
                if all_dl:
                    if fi["type"]=="xml":
                        up,err=update_xml(stt["file_content"],p["xpath"],{d["original_url"]:d["ftp_url"] for d in all_dl},p["new_node_name"],p["separator"])
                    else:
                        up,err=update_csv(stt["file_content"],p["column_name"],{d["original_url"]:d["ftp_url"] for d in all_dl},p["new_column_name"],p["separator"])
                    if err: st.error(err)
                    else:
                        b=up.encode(fi["encoding"]); st.session_state.output_bytes=b
                        ok,msg=save_to_drive(b,fi,{d["original_url"]:d["ftp_url"] for d in all_dl})
                        st.success("Zapisano na Drive" if ok else f"Błąd Drive: {msg}")
                        st.download_button("Pobierz",data=b,file_name=f"{os.path.splitext(fi['name'])[0]}_upd.{fi['type']}")
    with tab3:
        st.markdown("""
1. Wgraj plik XML lub CSV  
2. Podaj XPath/kolumnę i separator  
3. Uzupełnij dane FTP  
4. Pobierz i prześlij (proces w paczkach)  
5. Pobierz zaktualizowany plik  
6. Wznów przerwane przetwarzanie  
        """)

if __name__ == "__main__":
    main()
