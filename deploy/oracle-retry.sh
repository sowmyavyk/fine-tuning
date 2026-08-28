#!/bin/bash
# Oracle Always Free ARM capacity retry loop.
# Retries instance creation until a VM.Standard.A1.Flex slot frees up.
# Usage: ./oracle-retry.sh [AD-1|AD-2|AD-3]   (default AD-1)
set -e

AD="${1:-AD-1}"
REGION="ap-hyderabad-1"

echo "==> Fetching compartment, subnet, and image..."

COMPARTMENT=$(oci iam compartment list --compartment-id-in-subtree true --all \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print([c['id'] for c in d['data'] if c['lifecycle-state']=='ACTIVE'][0])")

# Root tenancy compartment (safest target)
TENANCY=$(oci iam tenancy get --tenancy $(oci iam compartment list 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['compartment-id'])") 2>/dev/null \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['id'])" 2>/dev/null) || COMPARTMENT=$COMPARTMENT

SUBNET=$(oci network subnet list --compartment-id "$COMPARTMENT" --all \
  | python3 -c "import json,sys; d=json.load(sys.stdin); subs=[s for s in d['data'] if 'public' in (s.get('display-name','').lower())]; print(subs[0]['id'] if subs else d['data'][0]['id'])")

IMAGE=$(oci compute image list --compartment-id "$COMPARTMENT" --shape VM.Standard.A1.Flex \
  --operating-system "Canonical Ubuntu" --sort-by TIMECREATED \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][-1]['id'])")

echo "compartment: $COMPARTMENT"
echo "subnet:      $SUBNET"
echo "image:       $IMAGE"
echo "target:      $AD / VM.Standard.A1.Flex (1 OCPU, 6 GB)"
echo "retrying every 45s until success (Ctrl-C to stop)..."
echo

for i in $(seq 1 200); do
  echo "attempt $i / 200 @ $(date '+%H:%M:%S')"
  if oci compute instance launch \
      --availability-domain "$AD" \
      --compartment-id "$COMPARTMENT" \
      --shape "VM.Standard.A1.Flex" \
      --shape-config '{"ocpus":1,"memoryInGBs":6}' \
      --image-id "$IMAGE" \
      --subnet-id "$SUBNET" \
      --ssh-authorized-keys-file ~/.ssh/oracle.pub \
      --display-name "finlens" \
      --assign-public-ip true \
      --wait-for-state RUNNING \
      --wait-interval-seconds 15 2>&1 | tail -1; then
    echo
    echo "SUCCESS! Instance created. Grab its public IP:"
    echo "  oci compute instance list --compartment-id $COMPARTMENT --display-name finlens"
    break
  fi
  echo "  (capacity not free — retrying in 45s)"
  sleep 45
done