# Wazuh Integration

## Custom Integration Requirements

Custom Wazuh integrations require **two files**:
1. A bash wrapper (e.g., `wazuh-integration/custom-aries`)
2. A Python script (e.g., `wazuh-integration/custom-aries.py`)

The bash wrapper must be executable and call the Python script explicitly.

## Python Interpreter

Wazuh containers have **no system `python3`**. Always use the embedded
interpreter:

```
/var/ossec/framework/python/bin/python3
```

## File Permissions

Scripts must be `750` with ownership `root:wazuh`.
`integratord` **rejects world-writable files** (permission error).

On NTFS bind mounts everything appears `777`. Use `docker cp` + `chmod`
instead of bind mounts:

```bash
docker cp wazuh-integration/custom-aries wazuh-manager:/var/ossec/integrations/
docker exec wazuh-manager chmod 750 /var/ossec/integrations/custom-aries
docker exec wazuh-manager chown root:wazuh /var/ossec/integrations/custom-aries
```

## ossec.conf Integration Block

```xml
<integration>
  <name>custom-aries</name>
  <level>3</level>
  <alert_format>json</alert_format>
</integration>
```

## Log Locations

- Custom integration output goes to **`ossec.log`**, NOT `integrations.log`.
- Enable debug logging:
  ```bash
  echo "integrator.debug=2" > /var/ossec/etc/local_internal_options.conf
  ```
- After `wazuh-control restart`, debug settings from
  `local_internal_options.conf` are preserved.

## Configuration File Mapping

Changing `wazuh_manager.conf` updates
`/wazuh-config-mount/etc/ossec.conf`.  
Restart the manager container to refresh the active
`/var/ossec/etc/ossec.conf`.

## Testing Tips

- FIM scans every 12 hours (`frequency=43200`); inject logs to
  `/var/log/auth.log` for quick iteration.
- Alerts JSONL format: use `tail -1 | python3 -m json.tool`, **not**
  `cat | python3 -m json.tool`.
