# Enterprise Policy & Contract Documents Repository

Place your organization's enterprise PDF documents here before running the ingestion pipeline.

### Expected Files:
- `employee_handbook.pdf`
- `leave_policy.pdf`
- `procurement_policy.pdf`
- `information_security_policy.pdf`
- `privacy_policy.pdf`
- `remote_work_policy.pdf`
- `vendor_contract.pdf`

### Automated Generation:
You can also generate realistic sample PDFs by running:
```bash
python scripts/generate_sample_docs.py
```

### Ingestion:
Once PDFs are present in this directory, run:
```bash
python -m backend.ingestion.ingest
```
