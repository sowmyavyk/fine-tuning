# FinLens on Oracle Cloud Always Free ARM

Run the full FinLens stack (Qwen2.5-1.5B + LoRA + RAG backend + Next.js
frontend + HTTPS) on Oracle Cloud's **Always Free ARM** instance — **$0/month**,
no HF subscription needed.

## 1. Create the instance (once, ~10 min)

1. Sign up at <https://signup.oraclecloud.com> — choose **"Always Free"** when
   asked about PAYG. **Do NOT upgrade to "Pay As You Go".**
2. Go to **Compute → Instances → Create instance**:
   - **Image**: `Canonical Ubuntu 22.04` (arm64)
   - **Shape**: `VM.Standard.A1.Flex` (Ampere ARM) — set **4 OCPU / 24 GB RAM**
     (this is within the free quota; Oracle shows it as the free max)
   - **Boot volume**: keep the free size (up to 200 GB free total)
   - **Networking**: default VCN/subnet; keep a **public IP** (ephemeral is fine)
   - **Add SSH key**: paste your public key (`ssh-keygen -t ed25519`)
3. Launch. Note the **public IP**.
4. Open ports (VCN → Security Lists → default → add ingress rules):
   - **TCP 80** and **TCP 443** (HTTPS via Caddy)
   - (optional) TCP 22 already open for SSH

## 2. Protect against surprise billing

1. **Never** click "Upgrade to PAYG" — always confirm you're on **Always Free**.
2. Set a budget alert: **Billing & Cost Management → Budgets → Create budget**,
   amount **$1**, alert at 100%. Oracle will email you if *anything* charges.
3. Only use free shapes/resources. If a bill appears anyway, open a support
   ticket with the usage report — free-tier overage is refundable.

## 3. Deploy the app

```bash
# from your Mac — copy the project to the VM once
scp -r deploy ubuntu@<PUBLIC_IP>:~

# ssh in
ssh ubuntu@<PUBLIC_IP>

# install docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# configure + run
cd ~/deploy
cp .env.example.oracle .env
# edit .env: set DOMAIN to your domain (or http://<PUBLIC_IP> to test first)
nano .env

docker compose -f docker-compose.oracle.yml up -d --build
```

First start downloads the base model (~3.3 GB) into the `hf-cache` volume, then
serves. Watch logs: `docker compose -f docker-compose.oracle.yml logs -f`.

## 4. Point a domain at it (for HTTPS)

### Option A — free subdomain (DuckDNS) — no purchase, works today
[DuckDNS](https://www.duckdns.org) gives you a free `yourname.duckdns.org`
subdomain and free HTTPS.

1. Go to <https://www.duckdns.org> and sign in with any account (GitHub/Google).
2. Add a subdomain, e.g. `finlens` → you get **`finlens.duckdns.org`**.
3. Set the **current IP** field to your VM's public IP and **Save**.
4. Make the subdomain follow the IP automatically (recommended). On the Oracle
   VM, set up a cron that updates DuckDNS every 5 min:
   ```bash
   ssh ubuntu@<PUBLIC_IP>
   sudo apt-get update && sudo apt-get install -y curl
   # replace <TOKEN> with your DuckDNS token (shown on the DuckDNS page)
   echo '*/5 * * * * curl -s "https://www.duckdns.org/update?domains=finlens&token=<TOKEN>&ip=" >> /tmp/duckdns.log' | sudo tee /etc/cron.d/duckdns
   ```
5. Configure the app:
   ```bash
   cd ~/deploy
   nano .env      # set DOMAIN=finlens.duckdns.org
   docker compose -f docker-compose.oracle.yml up -d
   ```
   Caddy auto-issues the Let's Encrypt cert for `finlens.duckdns.org`.
6. Clients open **https://finlens.duckdns.org** — a real, shareable HTTPS URL.

> DuckDNS IP updates are what make the ephemeral-public-IP (free tier) choice
> safe: even if Oracle reassigns the IP, the cron keeps the subdomain pointed
> at the current one.

### Option B — your own domain (if you already have one)
- In your DNS provider, add an **A record**: `finlens.example.com → <PUBLIC_IP>`.
- Set `DOMAIN=finlens.example.com` in `deploy/.env`, then:
  ```bash
  docker compose -f docker-compose.oracle.yml up -d   # Caddy auto-issues HTTPS
  ```
- Clients open **https://finlens.example.com**.

No domain yet and you just want to test quickly? Use `DOMAIN=http://<PUBLIC_IP>`
over plain HTTP first, then switch to DuckDNS for HTTPS.

## 5. Verify

```bash
curl -s http://localhost:3000/api/health        # expect {"status":"ok",...}
```

Then open the site and ask a compliance question. The backend streams tokens
over SSE with RBI citations, exactly like the local stack.

## On/off control

- **Stop/start**: Oracle console → the instance → Start / Stop. (Always Free
  ARM instances can be stopped to save nothing — they're free — but stopping
  is useful for maintenance.)
- **Pause the app** (keeps VM up): `docker compose -f docker-compose.oracle.yml stop`
- **Full teardown**: `docker compose -f docker-compose.oracle.yml down -v`

## Troubleshooting

- **`/api/health` empty or 502** → backend still downloading model; watch logs.
- **Ports not reachable** → VCN security list missing 80/443 ingress rules.
- **OOM during download** → the model is fp32 (~3.3 GB) but the VM has 24 GB;
  unlikely. If it happens, restart and the `hf-cache` volume resumes.
