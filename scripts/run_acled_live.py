from __future__ import annotations

from connectors.acled.connector import ACLEDConnector


def main() -> int:
    connector = ACLEDConnector(date_from="2024-04-01", date_to="2024-04-02")
    print("Starting ACLED incremental fetch (limited to 5 results)...")
    count = 0
    import traceback
    try:
        for rec in connector.fetch(mode="incremental"):
            print({"source_record_id": rec.get("source_record_id"), "source_url": rec.get("source_url")})
            count += 1
            if count >= 5:
                break
    except Exception as e:
        print("ERROR:")
        traceback.print_exc()
        return 1
    print(f"Fetched {count} records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
