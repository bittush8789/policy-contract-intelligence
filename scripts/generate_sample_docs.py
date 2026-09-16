"""Utility script to generate standard, valid sample enterprise policy and contract PDFs
for local testing of the ingestion, hybrid retrieval, and reranking pipelines.
Uses PyMuPDF (fitz) to create valid multi-page PDF files with headings and sections.
"""

import sys
from pathlib import Path
import fitz  # PyMuPDF

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"

SAMPLE_DATA = {
    "employee_handbook.pdf": [
        (
            "Section 1: Company Mission & Values",
            (
                "Welcome to Enterprise Corp. Our mission is to deliver world-class intelligence and enterprise services "
                "with the highest standards of integrity, innovation, and client satisfaction. All employees are expected "
                "to act with transparency, mutual respect, and accountability in all professional engagements."
            ),
        ),
        (
            "Section 2: Code of Conduct & Working Hours",
            (
                "Standard working hours are 40 hours per week, Monday through Friday, 9:00 AM to 5:00 PM local time, "
                "with a one-hour lunch period. Employees must treat colleagues, vendors, and clients with professional "
                "dignity. Harassment, discrimination, or abusive conduct of any kind is strictly prohibited."
            ),
        ),
        (
            "Section 3: Disciplinary Procedures",
            (
                "Failure to comply with company policies will result in disciplinary action up to and including termination. "
                "The disciplinary stages typically proceed as: verbal warning, formal written warning, performance "
                "improvement plan (30 days), and termination. In cases of gross misconduct, immediate termination may occur."
            ),
        ),
    ],
    "leave_policy.pdf": [
        (
            "Section 1: Annual Leave Entitlement",
            (
                "All full-time permanent employees are entitled to 25 days of paid annual leave per calendar year. "
                "Annual leave accrues on a monthly pro-rata basis from the employee's date of hire. Employees must submit "
                "leave requests via the HR portal at least two weeks in advance for absences exceeding three consecutive business days."
            ),
        ),
        (
            "Section 2: Sick & Medical Leave",
            (
                "Employees receive up to 10 days of paid sick leave per year for illness, medical appointments, or caring for "
                "an immediate family member. Any sick leave absence lasting longer than three consecutive business days "
                "requires a formal medical practitioner certificate submitted to HR upon return."
            ),
        ),
        (
            "Section 3: Leave Carry-Over Policy",
            (
                "Employees are strongly encouraged to take all accrued annual leave during the calendar year. A maximum of "
                "5 unused annual leave days may be carried over into the following year, and carried-over days must be utilized "
                "within the first quarter (by March 31). Unused days beyond 5 will be forfeited without financial compensation."
            ),
        ),
    ],
    "procurement_policy.pdf": [
        (
            "Section 1: Purpose & Authority Matrix",
            (
                "This policy establishes purchasing thresholds and required approvals for all corporate expenditures. "
                "All purchase requests must have an approved budget allocation prior to contract execution."
            ),
        ),
        (
            "Section 2: Approval Thresholds & Competitive Bidding",
            (
                "Approval thresholds are defined as follows: Purchases up to $10,000 require Department Manager approval. "
                "Purchases from $10,001 to $50,000 require Department Director approval and two competitive quotes. "
                "Purchases exceeding $50,000 require Vice President and Chief Financial Officer (CFO) approval along with "
                "a minimum of three competitive bids submitted to the Procurement Committee."
            ),
        ),
        (
            "Section 3: Vendor Vetting & Conflict of Interest",
            (
                "All new vendors must complete security, financial, and compliance vetting before onboarding. "
                "Employees must disclose any personal or financial relationship with prospective suppliers in writing to Legal."
            ),
        ),
    ],
    "information_security_policy.pdf": [
        (
            "Section 1: Password & Authentication Standards",
            (
                "All corporate accounts require passwords with a minimum length of 14 characters, combining uppercase letters, "
                "lowercase letters, numbers, and special symbols. Passwords must not contain user names or dictionary words. "
                "Multi-Factor Authentication (MFA) is strictly mandatory for all internal systems, VPN access, and cloud email."
            ),
        ),
        (
            "Section 2: Device Encryption & Clean Desk Policy",
            (
                "All company laptops, workstations, and mobile devices accessing corporate data must employ full-disk encryption "
                "using 256-bit AES (BitLocker for Windows, FileVault for macOS). Under the Clean Desk Policy, sensitive documents "
                "and portable storage media must be locked in drawers whenever workstations are unattended."
            ),
        ),
        (
            "Section 3: Security Incident Reporting",
            (
                "Any suspected security breach, unauthorized access, lost device, or phishing attempt must be reported immediately, "
                "and no later than one (1) hour following discovery, to the Security Operations Center via security@company.com."
            ),
        ),
    ],
    "privacy_policy.pdf": [
        (
            "Section 1: Data Protection Principles",
            (
                "Enterprise Corp processes personal data in strict compliance with the General Data Protection Regulation (GDPR), "
                "the California Consumer Privacy Act (CCPA), and applicable global privacy laws. Data collection is limited strictly "
                "to necessary business purposes."
            ),
        ),
        (
            "Section 2: Data Subject Rights & Fulfillment",
            (
                "Individuals have the right to request access, correction, data portability, or erasure of their personal information. "
                "All verified data subject access requests (DSARs) must be fulfilled by the Privacy Office within thirty (30) calendar days."
            ),
        ),
        (
            "Section 3: Data Retention & Secure Disposal",
            (
                "Customer and transaction records are retained for seven (7) years following contract termination to comply with statutory "
                "and audit obligations. Upon expiry of retention periods, records are permanently purged or securely shredded in accordance "
                "with NIST SP 800-88 guidelines."
            ),
        ),
    ],
    "remote_work_policy.pdf": [
        (
            "Section 1: Eligibility & Application",
            (
                "Regular full-time employees who have completed at least ninety (90) days of continuous service with satisfactory performance "
                "evaluations are eligible to apply for remote work arrangements. Remote work requests require written approval from both "
                "the Department Manager and the Human Resources Business Partner."
            ),
        ),
        (
            "Section 2: Working Hours & Core Availability",
            (
                "Remote employees must maintain regular working schedules and remain accessible via Slack, email, and video conferencing "
                "during mandatory core collaboration hours, defined as 10:00 AM to 4:00 PM local time."
            ),
        ),
        (
            "Section 3: Equipment & Home Office Stipend",
            (
                "The company provides a standard corporate laptop and peripherals. Eligible remote employees may claim a one-time "
                "home office setup stipend of up to $500 for ergonomic desk equipment upon submitting itemized receipts within 60 days of approval."
            ),
        ),
    ],
    "vendor_contract.pdf": [
        (
            "Section 1: Master Services Agreement & Scope",
            (
                "This Master Services Agreement ('Agreement') is entered into between Enterprise Corp ('Client') and "
                "Apex Cloud Solutions LLC ('Vendor'). Vendor agrees to provide cloud infrastructure management, 24/7 monitoring, "
                "and Tier-3 technical support services as detailed in Exhibit A."
            ),
        ),
        (
            "Section 2: Payment Terms & Invoicing",
            (
                "Vendor shall invoice Client on a monthly arrears basis. Client shall pay all undisputed invoice amounts within "
                "forty-five (45) days of receipt of invoice ('Net 45'). Late payments shall accrue interest at 1.0% per month."
            ),
        ),
        (
            "Section 3: Term & Termination Notice Period",
            (
                "This Agreement commences on the Effective Date for an initial term of two (2) years. Either party may terminate "
                "this Agreement without cause upon thirty (30) days prior written notice to the other party. In the event of material "
                "breach, the non-breaching party may terminate immediately if such breach remains uncured for fifteen (15) business days."
            ),
        ),
    ],
}


def generate_pdf(filepath: Path, pages_data: list):
    """Generate a clean, structured multi-page PDF using PyMuPDF."""
    doc = fitz.open()
    for page_idx, (section_title, section_text) in enumerate(pages_data, start=1):
        page = doc.new_page(width=595, height=842)  # Standard A4
        # Insert Header
        doc_title = filepath.stem.replace("_", " ").title()
        page.insert_text(
            (50, 60),
            f"ENTERPRISE CORP — {doc_title.upper()}",
            fontsize=11,
            color=(0.3, 0.3, 0.4),
        )
        page.draw_line((50, 70), (545, 70), color=(0.7, 0.7, 0.8), width=1)

        # Section Header
        page.insert_text(
            (50, 110),
            section_title,
            fontsize=15,
            color=(0.1, 0.15, 0.3),
        )

        # Body Paragraph
        rect = fitz.Rect(50, 140, 545, 750)
        page.insert_textbox(
            rect,
            section_text,
            fontsize=11,
            color=(0.1, 0.1, 0.1),
            lineheight=1.4,
        )

        # Footer
        page.draw_line((50, 790), (545, 790), color=(0.8, 0.8, 0.8), width=0.8)
        page.insert_text(
            (50, 808),
            f"Document: {filepath.name}  |  Page {page_idx}",
            fontsize=9,
            color=(0.5, 0.5, 0.5),
        )
        page.insert_text(
            (450, 808),
            "CONFIDENTIAL",
            fontsize=9,
            color=(0.7, 0.2, 0.2),
        )

    doc.save(str(filepath))
    doc.close()
    print(f"Generated valid PDF: {filepath.name} ({len(pages_data)} pages)")


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, pages in SAMPLE_DATA.items():
        pdf_path = DOCS_DIR / filename
        generate_pdf(pdf_path, pages)
    print("\nAll 7 enterprise sample PDFs successfully generated in data/documents/")


if __name__ == "__main__":
    main()
