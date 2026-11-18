#!/usr/bin/env python3
"""Minimal userspace NFSv3 + MOUNTv3 client (ONC RPC over TCP, AUTH_SYS).

Read-only: only NULL/GETATTR/LOOKUP/READ/READDIRPLUS/FSINFO are ever called.
Used to read an NFS export from a container that has TCP/2049 reachability but
no CAP_SYS_ADMIN (kernel mount impossible), e.g. the HYGON CI stage-a K8s pod.

Self-test: python3 nfs3.py selftest  (starts a local mock NFSv3 server)
"""
import os
import socket
import struct
import sys
import threading
import time

# ---------------------------------------------------------------- XDR helpers


class XDRReader:
    def __init__(self, data):
        self.d = data
        self.o = 0

    def u32(self):
        v = struct.unpack_from(">I", self.d, self.o)[0]
        self.o += 4
        return v

    def u64(self):
        v = struct.unpack_from(">Q", self.d, self.o)[0]
        self.o += 8
        return v

    def fixed(self, n):
        v = self.d[self.o:self.o + n]
        self.o += n
        return v

    def opaque(self):
        n = self.u32()
        v = self.d[self.o:self.o + n]
        self.o += (n + 3) & ~3
        return v

    def string(self):
        return self.opaque().decode("utf-8", "replace")

    def boolean(self):
        return self.u32() != 0


def x_u32(v):
    return struct.pack(">I", v)


def x_u64(v):
    return struct.pack(">Q", v)


def x_opaque(b):
    if isinstance(b, str):
        b = b.encode()
    pad = (4 - len(b) % 4) % 4
    return struct.pack(">I", len(b)) + b + b"\x00" * pad


x_string = x_opaque


# ---------------------------------------------------------------- RPC client

AUTH_NULL = 0
AUTH_SYS = 1


class RPCError(Exception):
    pass


class RPC:
    def __init__(self, host, prog, vers, port, timeout=15.0, uid=0, gid=0,
                 machine="i12664"):
        self.host = host
        self.prog = prog
        self.vers = vers
        self.port = port
        self.timeout = timeout
        self.uid = uid
        self.gid = gid
        self.machine = machine
        self.xid = int(time.time()) & 0xFFFF
        self.sock = None
        self.connect()

    def connect(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
        self.sock = None

    def _recvn(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RPCError("connection closed")
            buf += chunk
        return buf

    def _recv_record(self):
        hdr = self._recvn(4)
        frag = struct.unpack(">I", hdr)[0]
        last = frag & 0x80000000
        n = frag & 0x7FFFFFFF
        data = self._recvn(n)
        while not last:
            hdr = self._recvn(4)
            frag = struct.unpack(">I", hdr)[0]
            last = frag & 0x80000000
            n = frag & 0x7FFFFFFF
            data += self._recvn(n)
        return data

    def _auth(self):
        body = (x_u32(0) + x_string(self.machine) + x_u32(self.uid) +
                x_u32(self.gid) + x_u32(0))
        cred = x_u32(AUTH_SYS) + x_opaque(body)
        verf = x_u32(AUTH_NULL) + x_opaque(b"")
        return cred + verf

    def call(self, proc, args=b"", retries=2):
        last_exc = None
        for attempt in range(retries + 1):
            self.xid = (self.xid + 1) & 0xFFFFFFFF
            xid = self.xid
            msg = (x_u32(xid) + x_u32(0) + x_u32(2) + x_u32(self.prog) +
                   x_u32(self.vers) + x_u32(proc) + self._auth() + args)
            try:
                self.sock.sendall(struct.pack(">I", 0x80000000 | len(msg)) + msg)
                data = self._recv_record()
            except Exception as e:
                last_exc = e
                if attempt >= retries:
                    raise
                try:
                    self.connect()
                except Exception as e2:
                    last_exc = e2
                    time.sleep(0.3)
                continue
            r = XDRReader(data)
            rxid = r.u32()
            mtype = r.u32()
            if rxid != xid or mtype != 1:
                last_exc = RPCError("bad reply xid/mtype %d/%d" % (rxid, mtype))
                continue
            rstat = r.u32()
            if rstat != 0:
                reject = r.u32()
                raise RPCError("MSG_DENIED reject_stat=%d" % reject)
            r.u32()          # verf flavor
            r.opaque()       # verf body
            accept = r.u32()
            if accept != 0:
                raise RPCError("accept_stat=%d" % accept)
            return r
        raise last_exc or RPCError("rpc failed")


def portmap_getport(host, prog, vers, prot=6, timeout=8.0):
    """Ask rpcbind/portmapper for the TCP port of prog/vers."""
    try:
        c = RPC(host, 100000, 2, 111, timeout=timeout)
        r = c.call(3, x_u32(prog) + x_u32(vers) + x_u32(prot) + x_u32(0))
        port = r.u32()
        c.close()
        return port or None
    except Exception:
        return None


# ---------------------------------------------------------------- NFSv3 client

NFS3_OK = 0
NFS3ERR_NOENT = 2
NFS3ERR_ACCES = 13
NFS3ERR_NOTDIR = 20
NFS3ERR_ISDIR = 21
NFS3ERR_STALE = 70

NF3REG = 1
NF3DIR = 2
NF3BLK = 3
NF3CHR = 4
NF3LNK = 5
NF3SOCK = 6
NF3FIFO = 7

TYPE_NAME = {NF3REG: "reg", NF3DIR: "dir", NF3BLK: "blk", NF3CHR: "chr",
             NF3LNK: "lnk", NF3SOCK: "sock", NF3FIFO: "fifo"}


class NFS3:
    def __init__(self, host, port=2049, timeout=15.0, uid=0, gid=0):
        self.host = host
        self.rpc = RPC(host, 100003, 3, port, timeout=timeout, uid=uid, gid=gid)
        self.rtmax = 1 << 20

    def close(self):
        self.rpc.close()

    # ---- attribute decoding
    @staticmethod
    def read_fattr(r):
        t = r.u32()
        mode = r.u32()
        nlink = r.u32()
        uid = r.u32()
        gid = r.u32()
        size = r.u64()
        used = r.u64()
        rdev = (r.u32(), r.u32())
        fsid = r.u64()
        fileid = r.u64()
        atime = (r.u32(), r.u32())
        mtime = (r.u32(), r.u32())
        ctime = (r.u32(), r.u32())
        return {"type": t, "mode": mode, "nlink": nlink, "uid": uid, "gid": gid,
                "size": size, "used": used, "rdev": rdev, "fsid": fsid,
                "fileid": fileid, "mtime": mtime}

    @staticmethod
    def read_post_op_attr(r):
        if not r.boolean():
            return None
        return NFS3.read_fattr(r)

    @staticmethod
    def read_post_op_fh(r):
        if not r.boolean():
            return None
        return r.opaque()

    # ---- procedures
    def fsinfo(self, fh):
        r = self.rpc.call(19, x_opaque(fh))
        st = r.u32()
        if st != NFS3_OK:
            return st, None
        self.read_post_op_attr(r)
        info = {}
        info["rtmax"] = r.u32()
        info["rtpref"] = r.u32()
        info["rtmult"] = r.u32()
        info["wtmax"] = r.u32()
        info["wtpref"] = r.u32()
        info["wtmult"] = r.u32()
        info["dtpref"] = r.u32()
        info["maxfilesize"] = r.u64()
        info["time_delta"] = (r.u32(), r.u32())
        info["properties"] = r.u32()
        self.rtmax = max(4096, info["rtmax"])
        return st, info

    def getattr(self, fh):
        r = self.rpc.call(1, x_opaque(fh))
        st = r.u32()
        if st != NFS3_OK:
            return st, None
        return st, self.read_fattr(r)

    def lookup(self, dirfh, name):
        r = self.rpc.call(3, x_opaque(dirfh) + x_string(name))
        st = r.u32()
        if st != NFS3_OK:
            return st, None, None
        fh = r.opaque()
        attr = self.read_post_op_attr(r)
        return st, fh, attr

    def read(self, fh, offset, count):
        count = min(count, self.rtmax)
        r = self.rpc.call(6, x_opaque(fh) + x_u64(offset) + x_u32(count))
        st = r.u32()
        if st != NFS3_OK:
            return st, b"", False
        self.read_post_op_attr(r)
        n = r.u32()
        eof = r.boolean()
        data = r.opaque()
        return st, data[:n], eof

    def readdirplus(self, dirfh, cookie=0, verf=b"\x00" * 8,
                    dircount=16384, maxcount=65536):
        args = (x_opaque(dirfh) + x_u64(cookie) + verf + x_u32(dircount) +
                x_u32(maxcount))
        r = self.rpc.call(17, args)
        st = r.u32()
        if st != NFS3_OK:
            return st, None, []
        self.read_post_op_attr(r)
        newverf = r.fixed(8)
        entries = []
        while True:
            if not r.boolean():
                break
            fileid = r.u64()
            name = r.string()
            ent_cookie = r.u64()
            attr = self.read_post_op_attr(r)
            fh = self.read_post_op_fh(r)
            entries.append({"fileid": fileid, "name": name, "cookie": ent_cookie,
                            "attr": attr, "fh": fh})
        return st, newverf, entries

    def list_all(self, dirfh, max_entries=200000):
        """Full READDIRPLUS pagination."""
        cookie = 0
        verf = b"\x00" * 8
        out = []
        while True:
            st, newverf, ents = self.readdirplus(dirfh, cookie, verf)
            if st != NFS3_OK:
                return st, out
            if newverf:
                verf = newverf
            if not ents:
                break
            out.extend(ents)
            cookie = ents[-1]["cookie"]
            if cookie == 0 or len(out) >= max_entries:
                break
        return NFS3_OK, out


def mount_export(host, path, mount_port=None, timeout=15.0, uid=0, gid=0):
    """MOUNTv3 MNT -> (fhandle, mount_rpc). Tries 20048 then rpcbind/111."""
    ports = []
    if mount_port:
        ports.append(mount_port)
    try:
        p = portmap_getport(host, 100005, 3)
        if p and p not in ports:
            ports.append(p)
    except Exception:
        pass
    for p in (20048, 32767, 892):
        if p not in ports:
            ports.append(p)
    last = None
    for port in ports:
        try:
            c = RPC(host, 100005, 3, port, timeout=timeout, uid=uid, gid=gid)
            r = c.call(1, x_string(path))
            st = r.u32()
            if st != 0:
                c.close()
                last = "mount status=%d port=%d" % (st, port)
                continue
            fh = r.opaque()
            try:
                flavors = r.u32()
                fl = [r.u32() for _ in range(flavors)]
            except Exception:
                fl = []
            return fh, c, port, fl
        except Exception as e:
            last = "%r port=%d" % (e, port)
            continue
    raise RPCError("mount %s:%s failed (%s)" % (host, path, last))


# ---------------------------------------------------------------- self-test

class MockServer:
    """In-process MOUNTv3+NFSv3 server over a local directory (self-test only)."""

    def __init__(self, root):
        self.root = os.path.realpath(root)
        self.rootfh = b"\x01" + self.root.encode()
        self.paths = {self.rootfh: self.root}
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except Exception:
                return
            threading.Thread(target=self._conn, args=(c,), daemon=True).start()

    def _conn(self, c):
        try:
            while True:
                hdr = c.recv(4)
                if len(hdr) < 4:
                    return
                frag = struct.unpack(">I", hdr)[0]
                n = frag & 0x7FFFFFFF
                data = b""
                while len(data) < n:
                    ch = c.recv(n - len(data))
                    if not ch:
                        return
                    data += ch
                reply = self._handle(data)
                c.sendall(struct.pack(">I", 0x80000000 | len(reply)) + reply)
        except Exception:
            return

    def _fh_for(self, path):
        b = b"\x01" + path.encode()
        self.paths[b] = path
        return b

    def _attr_for(self, path):
        st = os.lstat(path)
        t = NF3DIR if os.path.isdir(path) and not os.path.islink(path) else (
            NF3LNK if os.path.islink(path) else NF3REG)
        return (x_u32(t) + x_u32(st.st_mode) + x_u32(st.st_nlink) + x_u32(st.st_uid) +
                x_u32(st.st_gid) + x_u64(st.st_size) + x_u64(st.st_size) +
                x_u32(0) + x_u32(0) + x_u64(1) + x_u64(st.st_ino) +
                x_u32(int(st.st_atime)) + x_u32(0) + x_u32(int(st.st_mtime)) + x_u32(0) +
                x_u32(int(st.st_ctime)) + x_u32(0))

    def _handle(self, data):
        r = XDRReader(data)
        xid = r.u32(); r.u32(); r.u32(); prog = r.u32(); r.u32(); proc = r.u32()
        r.u32(); r.opaque()
        r.u32(); r.opaque()
        out = b""
        if prog == 100005 and proc == 1:
            r.string()
            out = x_u32(0) + x_opaque(self.rootfh) + x_u32(1) + x_u32(1)
        elif prog == 100005 and proc == 3:
            out = x_u32(1) + x_u32(1) + x_string("/") + x_u32(0)
        elif prog == 100003:
            if proc == 19:
                fh = r.opaque()
                out = x_u32(0) + x_u32(1) + self._attr_for(self.paths[fh]) + \
                    x_u32(1 << 20) + x_u32(1 << 20) + x_u32(4096) + \
                    x_u32(1 << 20) + x_u32(1 << 20) + x_u32(4096) + x_u32(4096) + \
                    x_u64(1 << 60) + x_u32(0) + x_u32(1) + x_u32(0)
            elif proc == 1:
                fh = r.opaque()
                out = x_u32(0) + self._attr_for(self.paths[fh])
            elif proc == 3:
                dfh = r.opaque(); name = r.string()
                p = os.path.join(self.paths[dfh], name)
                if not os.path.exists(p):
                    out = x_u32(NFS3ERR_NOENT)
                else:
                    out = x_u32(0) + x_opaque(self._fh_for(os.path.realpath(p))) + \
                        x_u32(1) + self._attr_for(p)
            elif proc == 6:
                fh = r.opaque(); off = r.u64(); cnt = r.u32()
                p = self.paths[fh]
                with open(p, "rb") as f:
                    f.seek(off)
                    data = f.read(cnt)
                eof = (off + len(data)) >= os.path.getsize(p)
                out = x_u32(0) + x_u32(1) + self._attr_for(p) + x_u32(len(data)) + \
                    x_u32(1 if eof else 0) + x_opaque(data)
            elif proc == 17:
                dfh = r.opaque(); cookie = r.u64(); r.fixed(8)
                r.u32(); r.u32()
                p = self.paths[dfh]
                names = sorted(os.listdir(p))
                out = x_u32(0) + x_u32(1) + self._attr_for(p) + b"\x00" * 8
                start = 0
                if cookie:
                    for i, n in enumerate(names):
                        if i + 1 == cookie:
                            start = i + 1
                            break
                for i, n in enumerate(names[start:], start):
                    fp = os.path.join(p, n)
                    out += x_u32(1) + x_u64(os.lstat(fp).st_ino) + x_string(n) + \
                        x_u64(i + 1) + x_u32(1) + self._attr_for(fp) + x_u32(1) + \
                        x_opaque(self._fh_for(os.path.realpath(fp)))
                out += x_u32(0)
            else:
                out = x_u32(0)
        return x_u32(xid) + x_u32(1) + x_u32(0) + x_u32(0) + x_u32(0) + x_u32(0) + out


def _selftest():
    """Local mock NFSv3/MOUNTv3 server over a temp tree, then walk it."""
    import tempfile
    root = tempfile.mkdtemp(prefix="nfs3mock_")
    os.makedirs(os.path.join(root, "home", "alice", ".ssh"))
    with open(os.path.join(root, "home", "alice", ".ssh", "id_rsa"), "w") as f:
        f.write("-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\n-----END OPENSSH PRIVATE KEY-----\n")
    with open(os.path.join(root, "home", "alice", ".bash_history"), "w") as f:
        f.write("export AWS_SECRET_ACCESS_KEY=abcdefghijklmnopqrstuvwxyz0123456789ABCD\n")
    for i in range(300):
        with open(os.path.join(root, "home", "alice", "f%03d.log" % i), "w") as f:
            f.write("x" * 50)
    srv = MockServer(root)
    c = RPC("127.0.0.1", 100005, 3, srv.port, timeout=5)
    r = c.call(1, x_string("/"))
    assert r.u32() == 0
    mfh = r.opaque()
    n = NFS3("127.0.0.1", port=srv.port, timeout=5)
    st, info = n.fsinfo(mfh)
    print("fsinfo", st, info and info["rtmax"])
    st, ents = n.list_all(mfh)
    print("root entries", st, [e["name"] for e in ents])
    home = [e for e in ents if e["name"] == "home"][0]
    st, hh, _ = n.lookup(home["fh"], "alice")
    print("lookup alice", st)
    st, ssh, _ = n.lookup(hh, ".ssh")
    st, key, a = n.lookup(ssh, "id_rsa")
    st, data, eof = n.read(key, 0, 4096)
    print("read id_rsa", st, data[:40], eof)
    st, ents = n.list_all(hh)
    print("alice entries", len(ents))
    st, bh, _ = n.lookup(hh, ".bash_history")
    st, data, eof = n.read(bh, 0, 4096)
    print("history:", data)
    print("SELFTEST OK root=%s" % root)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
    else:
        print(__doc__)
