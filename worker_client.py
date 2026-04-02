"""
Distributed SMTP verification via SSH.

No API, no Flask, no setup on the VPS - just SSH + Python.
Runs SMTP checks on remote machines via SSH, rotates between them.

Config in ~/.x_pathfinder/workers.json:
[
    {"host": "vps1.example.com", "user": "root"},
    {"host": "vps2.example.com", "user": "root", "port": 2222},
    {"host": "vps3.example.com", "user": "root", "key": "~/.ssh/vps3_key"}
]
"""

import os
import json
import random
import asyncio
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

WORKERS_FILE = os.path.expanduser("~/.x_pathfinder/workers.json")

# Python one-liner that runs on the remote VPS - no dependencies needed
REMOTE_SCRIPT = '''
import smtplib,sys,random,string
email=sys.argv[1]
domain=email.split("@")[1]
def mx(d):
    try:
        import dns.resolver
        return str(dns.resolver.resolve(d,"MX")[0].exchange).rstrip(".")
    except:return d
def chk(e,d):
    try:
        s=smtplib.SMTP(timeout=10);s.connect(mx(d),25);s.ehlo("v.local");s.mail("v@v.local")
        c,_=s.rcpt(e);s.quit();return c
    except:return -1
# catch-all check
r=''.join(random.choices(string.ascii_lowercase,k=20))
fake_code=chk(f"xpf_{r}@{domain}",domain)
if fake_code==250:print(f"CATCHALL");sys.exit()
code=chk(email,domain)
if code==250:print("EXISTS")
elif code==550:print("REJECTED")
else:print(f"UNKNOWN:{code}")
'''


class WorkerClient:
    """Distributed SMTP verification via SSH."""

    def __init__(self):
        self.workers = self._load_workers()
        self._healthy: List[Dict] = list(self.workers)
        self._catchall_domains: set = set()

    def _load_workers(self) -> List[Dict]:
        path = Path(WORKERS_FILE)
        if path.exists():
            try:
                with open(path) as f:
                    workers = json.load(f)
                logger.info(f"Loaded {len(workers)} SSH workers")
                return workers
            except Exception as e:
                logger.warning(f"Failed to load workers: {e}")
        return []

    @property
    def available(self) -> bool:
        return len(self._healthy) > 0

    def _pick_worker(self) -> Optional[Dict]:
        if not self._healthy:
            self._healthy = list(self.workers)
        if not self._healthy:
            return None
        return random.choice(self._healthy)

    def _mark_unhealthy(self, worker: Dict):
        if worker in self._healthy:
            self._healthy.remove(worker)
            logger.warning(
                f"Worker {worker['host']} down "
                f"({len(self._healthy)}/{len(self.workers)} left)"
            )

    def _build_ssh_cmd(self, worker: Dict, email: str) -> List[str]:
        """Build SSH command to run SMTP check on remote VPS."""
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5"]

        if worker.get("port"):
            cmd.extend(["-p", str(worker["port"])])
        if worker.get("key"):
            cmd.extend(["-i", os.path.expanduser(worker["key"])])

        user_host = f"{worker.get('user', 'root')}@{worker['host']}"
        cmd.append(user_host)

        # Run python one-liner on remote
        cmd.extend(["python3", "-c", REMOTE_SCRIPT, email])
        return cmd

    async def verify_email(self, email: str) -> Optional[Dict]:
        """Verify email via SSH to random worker."""
        domain = email.split("@")[1]

        # Skip known catch-all domains
        if domain in self._catchall_domains:
            return {"email": email, "exists": None, "catch_all": True, "code": -1}

        worker = self._pick_worker()
        if not worker:
            return None

        try:
            cmd = self._build_ssh_cmd(worker, email)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=20
            )
            output = stdout.decode().strip()

            if output == "EXISTS":
                return {"email": email, "exists": True, "catch_all": False, "code": 250}
            elif output == "REJECTED":
                return {"email": email, "exists": False, "catch_all": False, "code": 550}
            elif output == "CATCHALL":
                self._catchall_domains.add(domain)
                return {"email": email, "exists": None, "catch_all": True, "code": -1}
            else:
                return {"email": email, "exists": None, "catch_all": False, "code": -1}

        except asyncio.TimeoutError:
            self._mark_unhealthy(worker)
            return None
        except Exception as e:
            self._mark_unhealthy(worker)
            logger.debug(f"SSH worker {worker['host']} failed: {e}")
            return None

    async def verify_batch(self, emails: List[str]) -> List[Dict]:
        """Verify batch, distributed across workers."""
        tasks = [self.verify_email(e) for e in emails]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]

    async def health_check(self) -> Dict[str, bool]:
        """Check which workers are reachable via SSH."""
        status = {}
        for worker in self.workers:
            try:
                cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
                       "-o", "ConnectTimeout=3"]
                if worker.get("port"):
                    cmd.extend(["-p", str(worker["port"])])
                if worker.get("key"):
                    cmd.extend(["-i", os.path.expanduser(worker["key"])])

                user_host = f"{worker.get('user', 'root')}@{worker['host']}"
                cmd.extend([user_host, "echo ok"])

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                status[worker["host"]] = stdout.decode().strip() == "ok"
            except Exception:
                status[worker["host"]] = False

        self._healthy = [
            w for w in self.workers if status.get(w["host"], False)
        ]
        return status

    async def close(self):
        pass  # Nothing to clean up for SSH
