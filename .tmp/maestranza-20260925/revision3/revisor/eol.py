import subprocess
for f in ["README.md","CHANGELOG.md"]:
    new=open(f,'rb').read()
    old=subprocess.run(["git","show","HEAD:"+f],capture_output=True).stdout
    print(f, "HEAD crlf/lf", old.count(b"\r\n"), old.count(b"\n"), "NOW", new.count(b"\r\n"), new.count(b"\n"))
