#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ARIES — Docker Storage Cleanup Script
# ═══════════════════════════════════════════════════════════════════════
#
# Frees disk space on the Linux root partition by:
#   1. Removing orphaned Docker volumes (anonymous, unused)
#   2. Pruning Docker build cache
#   3. Removing unused Docker images
#   4. Showing remaining disk usage
#
# Usage:
#   bash MLOps/cleanup_docker.sh          # interactive (asks before each step)
#   bash MLOps/cleanup_docker.sh --all    # non-interactive (cleans everything)
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

AUTO=false
[[ "${1:-}" == "--all" ]] && AUTO=true

echo -e "\n${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ARIES — Docker Storage Cleanup${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}\n"

echo -e "${YELLOW}Current disk usage:${NC}"
df -h / /data 2>/dev/null | grep -E 'Filesystem|/'
echo ""

echo -e "${YELLOW}Docker disk breakdown:${NC}"
docker system df 2>/dev/null
echo ""

# ── 1. Orphaned volumes ─────────────────────────────────────────────

# Named volumes belonging to Aries that now use bind mounts
ARIES_OLD_VOLUMES=(
    mlops_minio_data
    mlops_postgres_data
    mlops_pgadmin_data
    fastapi_service_kafka_data
    fastapi_service_redis_data
    aries_minio_data
    aries_pg_data
    aries_pgadmin_data
    aries_kafka_data
    aries_redis_data
)

echo -e "${YELLOW}Step 1: Remove old named ARIES volumes (now using bind mounts on /data)${NC}"
for vol in "${ARIES_OLD_VOLUMES[@]}"; do
    if docker volume inspect "$vol" &>/dev/null; then
        echo -e "  Found: ${RED}$vol${NC}"
        if $AUTO || (read -rp "  Delete $vol? [y/N] " ans && [[ "$ans" =~ ^[Yy] ]]); then
            docker volume rm "$vol" 2>/dev/null && echo -e "    ${GREEN}Removed${NC}" || echo -e "    ${RED}In use — stop containers first${NC}"
        fi
    fi
done

echo ""
echo -e "${YELLOW}Step 2: Remove anonymous (orphaned) volumes${NC}"
ANON_VOLS=$(docker volume ls -q --filter "dangling=true" | grep -vE '^(mlops_|fastapi_|aries_|single-node_|n8n_)' || true)
if [[ -n "$ANON_VOLS" ]]; then
    echo "  Found $(echo "$ANON_VOLS" | wc -l) anonymous volume(s)"
    if $AUTO || (read -rp "  Delete all anonymous volumes? [y/N] " ans && [[ "$ans" =~ ^[Yy] ]]); then
        echo "$ANON_VOLS" | xargs docker volume rm 2>/dev/null && echo -e "    ${GREEN}Removed${NC}" || echo -e "    ${RED}Some volumes in use${NC}"
    fi
else
    echo "  No anonymous orphaned volumes found"
fi

# ── 2. Build cache ──────────────────────────────────────────────────

echo ""
echo -e "${YELLOW}Step 3: Prune Docker build cache${NC}"
BUILD_CACHE=$(docker system df --format '{{.Size}}' 2>/dev/null | tail -1)
echo "  Build cache size: $BUILD_CACHE"
if $AUTO || (read -rp "  Prune all build cache? [y/N] " ans && [[ "$ans" =~ ^[Yy] ]]); then
    docker builder prune -af 2>/dev/null && echo -e "    ${GREEN}Pruned${NC}"
fi

# ── 3. Unused images ────────────────────────────────────────────────

echo ""
echo -e "${YELLOW}Step 4: Remove unused Docker images${NC}"
UNUSED=$(docker images -q --filter "dangling=true" 2>/dev/null | wc -l)
ALL_UNUSED=$(docker images -q 2>/dev/null | wc -l)
echo "  Dangling images: $UNUSED,  Total images: $ALL_UNUSED"
if $AUTO || (read -rp "  Remove all unused images? [y/N] " ans && [[ "$ans" =~ ^[Yy] ]]); then
    docker image prune -af 2>/dev/null && echo -e "    ${GREEN}Pruned${NC}"
fi

# ── 4. Stopped containers ──────────────────────────────────────────

echo ""
echo -e "${YELLOW}Step 5: Remove stopped containers${NC}"
STOPPED=$(docker ps -aq --filter "status=exited" 2>/dev/null | wc -l)
echo "  Stopped containers: $STOPPED"
if [[ "$STOPPED" -gt 0 ]]; then
    docker ps -a --filter "status=exited" --format "  {{.Names}} ({{.Image}}) — {{.Status}}"
    if $AUTO || (read -rp "  Remove all stopped containers? [y/N] " ans && [[ "$ans" =~ ^[Yy] ]]); then
        docker container prune -f 2>/dev/null && echo -e "    ${GREEN}Pruned${NC}"
    fi
fi

# ── Summary ─────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  After cleanup:${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
df -h / /data 2>/dev/null | grep -E 'Filesystem|/'
echo ""
docker system df 2>/dev/null
