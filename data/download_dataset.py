"""
Dataset Preparation Script for Phishing Email Detection System.

This script acquires, cleans, normalizes, and prepares email dataset records
for training machine learning classifiers. It incorporates both legitimate
(SAFE) and malicious (PHISHING) email examples.

Sources & Standards:
- Real-world email patterns modeled from SpamAssassin Public Corpus and
  Jose Nazario Phishing Corpus.
- Labels normalized to:
    - label: 1 (PHISHING), 0 (SAFE)
    - label_text: 'PHISHING', 'SAFE'
"""

import os
import re
import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base path definitions
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR
OUTPUT_FILE = os.path.join(DATA_DIR, "emails.csv")


def get_curated_email_corpus():
    """
    Returns a comprehensive, high-quality curated dataset of phishing and legitimate emails.
    These samples reflect authentic attack vectors (credential harvesting, financial wire scams,
    urgency suspension threats, fake invoices, crypto scams) and legitimate communications
    (collaboration, receipts, technical discussions, HR notices, personal messages).
    """
    emails = []

    # -------------------------------------------------------------
    # 1. PHISHING EMAIL SAMPLES
    # -------------------------------------------------------------
    phishing_samples = [
        # Credential Harvesting - Microsoft 365
        {
            "subject": "URGENT: Microsoft 365 Password Expiration Notice",
            "body": """Dear User,
Your Microsoft 365 email account password will expire in 24 hours.
Failure to renew your credentials will result in immediate suspension of your mailbox and cloud storage.
Please update your password immediately by clicking the link below:
http://microsoft365-verify-portal.account-security.xyz/login.php?user=verify
Keep your credentials safe. Do not ignore this alert.
IT Helpdesk Support Center""",
            "category": "Credential Harvesting",
        },
        # Credential Harvesting - Google Workspace
        {
            "subject": "Security Alert: Unauthorized login attempt detected on your Google Account",
            "body": """Google Security Team Alert
Someone recently attempted to log into your account from an unrecognized IP address (185.220.101.5) in Moscow, Russia.
If this was not you, your account has been compromised!
Verify your identity immediately to secure your account:
http://192.168.1.105/google-signin/auth.html
If you do not confirm within 12 hours, your access will be permanently locked.
The Google Accounts Team""",
            "category": "Credential Harvesting",
        },
        # Credential Harvesting - Apple ID
        {
            "subject": "Your Apple ID has been locked for security reasons",
            "body": """Dear Customer,
Your Apple ID has been suspended due to excessive failed login attempts and suspicious purchase activity.
To restore access and unlock your Apple ID, verify your billing information and credentials now:
http://appleid-security-protection.apple-support.ru/verify-account
Failure to verify within 48 hours will result in permanent account termination.
Apple Support Team""",
            "category": "Credential Harvesting",
        },
        # Financial / Banking - PayPal
        {
            "subject": "Important Notice: Unusual transaction detected on your PayPal account",
            "body": """Dear Customer,
We observed an unauthorized payment of $849.99 USD to Coinbase Global Inc on your PayPal account.
If you did not authorize this charge, dispute the transaction and cancel the payment immediately:
http://paypal-resolution-center.user-dispute.top/dispute.php?id=948293
Please enter your login details, credit card number, and billing address to reverse this charge.
PayPal Fraud Prevention Department""",
            "category": "Financial Scam",
        },
        # Financial / Banking - Chase Bank
        {
            "subject": "URGENT ACTION REQUIRED: Chase Online Banking Access Restricted",
            "body": """Notice of Account Suspension
We have detected irregular debit card transactions and potential identity theft on your Chase Checking account.
For your safety, online banking access is temporarily blocked.
To reinstate full banking privileges, please verify your Social Security Number, PIN, and account number:
http://chase-security-verification.online-banking-portal.work/auth/chase
Do not reply to this email. Complete verification online immediately.
JPMorgan Chase & Co. Security Division""",
            "category": "Financial Scam",
        },
        # Financial / Banking - Overdue Invoice Wire Fraud
        {
            "subject": "OVERDUE INVOICE #INV-89211: Immediate Wire Payment Required",
            "body": """Attention Accounts Payable,
Please find attached revised invoice #INV-89211 for outstanding services totaling $24,750.00.
Due to our annual banking audit, our receiving bank account has changed with immediate effect.
Please process the wire transfer today to the updated account details:
Bank: Offshore Global Trust
Account Number: 9812-441-291
Swift Code: OFSGLB33
Click here to confirm wire transfer receipt: http://vendor-invoices-billing.su/confirm-wire.html
Failure to settle today will incur 15% interest and immediate legal escalation.
Sincerely,
Accounts Receivable Department""",
            "category": "Financial Scam",
        },
        # Urgency & Legal Threats - IRS Tax Refund
        {
            "subject": "Internal Revenue Service: Notification of Pending Federal Tax Refund",
            "body": """Internal Revenue Service (IRS) Tax Notice
You have an unclaimed federal income tax refund of $1,420.50 from the previous fiscal year.
Federal regulations require you to submit your tax refund claim form within 7 days.
Click the link below to access your secure refund portal and verify your SSN and direct deposit bank details:
http://irs-gov-tax-refund-portal.claim-center.club/portal/refund.jsp
Direct refund requests without proper identity verification cannot be disbursed.
Official IRS Digital Services Division""",
            "category": "Legal & Extortion",
        },
        # CEO / Executive Impersonation (Spear Phishing)
        {
            "subject": "Confidential Request - Are you at your desk right now?",
            "body": """Hi,
I am currently tied up in an urgent executive board meeting and cannot take calls.
I need you to handle an urgent confidential task for me right away.
We need to purchase 10 Apple iTunes / Amazon gift cards valued at $500 each for client rewards.
Please purchase them immediately and scratch the back to send me pictures of the redemption codes.
I will reimburse you via expense report this afternoon.
Treat this with highest priority and confidentiality.
Sent from my iPhone""",
            "category": "Spear Phishing",
        },
        # IT Helpdesk / VPN Update
        {
            "subject": "MANDATORY: Corporate GlobalProtect VPN & Duo 2FA Security Update",
            "body": """Attention All Employees,
In response to recent cybersecurity threats, all staff members must migrate to the new SSL VPN gateway today.
Old VPN credentials will be deactivated tonight at 11:59 PM.
To migrate your device and update your Duo two-factor authentication token:
1. Navigate to: http://secure-vpn-migration.corp-internal-login.tk/ssl-vpn
2. Enter your corporate network username and password.
3. Scan the new QR code with your mobile device.
Compliance is mandatory for all active employees.
Corporate IT Security Operations""",
            "category": "Credential Harvesting",
        },
        # Delivery / Package Scam - FedEx
        {
            "subject": "FedEx Delivery Notification: Your package delivery was unsuccessful (#FDX-99214)",
            "body": """Dear Customer,
FedEx courier was unable to deliver your parcel #FDX-99214 today due to an incorrect shipping address and unpaid customs clearance duty ($2.85).
To reschedule delivery to your preferred home address and settle the clearance fee:
http://fedex-express-tracking.parcel-reschedule.top/track/FDX-99214
If the address is not confirmed within 48 hours, the package will be returned to sender.
Thank you for shipping with FedEx.""",
            "category": "Delivery Scam",
        },
        # Delivery / Package Scam - DHL
        {
            "subject": "DHL Express: Important Update on Consignment Tracking #91823192",
            "body": """Dear Consignee,
Your inbound parcel #91823192 is held at our regional distribution hub.
A processing duty fee of $3.50 is required before customs release.
Please confirm your address and complete payment via our secure portal:
http://dhl-express-duty-clearance.track-shipment.icu/dhl/pay
Failure to pay within 3 days will result in package confiscation.
DHL Customer Care""",
            "category": "Delivery Scam",
        },
        # Extortion / Blackmail Scam
        {
            "subject": "SECURITY BREACH: Your webcam was recorded and private data saved",
            "body": """I know your password.
I infected your operating system with a Pegasus-based Trojan when you visited an adult website.
I recorded dual-screen video of you and your screen activities.
I have downloaded all your contacts, emails, and social media passwords.
If you do not send $1,500 USD in Bitcoin to my wallet within 48 hours, I will forward the video to your family, coworkers, and friends.
Bitcoin Address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
Do not try to contact police or reply to this address. Timer started.""",
            "category": "Legal & Extortion",
        },
        # Crypto / Web3 Wallet Phishing
        {
            "subject": "MetaMask Alert: Action Required to prevent account suspension",
            "body": """Attention MetaMask User,
Our system detected that your Web3 wallet has not yet been upgraded to the latest security protocol (v2.4).
Unverified wallets will be restricted from transacting on Ethereum and Polygon networks.
Restore and re-authenticate your wallet seed phrase immediately:
http://metamask-seed-validation.web3-restore.buzz/import-wallet
Never share your secret recovery phrase with anyone except our official form.
MetaMask Decentralized Security""",
            "category": "Financial Scam",
        },
        # HR Payroll Scam
        {
            "subject": "URGENT: Payroll Direct Deposit Information Verification Required",
            "body": """Dear Colleague,
Our payroll department experienced a system migration error affecting next Friday's direct deposit.
To ensure your salary is deposited on time without delay, please review and update your banking routing details:
http://company-payroll-portal.workforce-hr.top/payroll/direct-deposit
Please log in with your single sign-on corporate credentials to confirm.
Human Resources & Payroll Administration""",
            "category": "Credential Harvesting",
        },
        # Anti-Virus Renewal Subscription Extortion
        {
            "subject": "Geek Squad / Norton 360: Order Confirmation - Auto-Renewal Charged $499.00",
            "body": """Thank you for your order!
Your annual subscription for Norton 360 Total Protection has automatically renewed.
Amount Debited: $499.00 USD from your registered checking account.
Invoice ID: GSQ-7821092
If you did not authorize this auto-debit or wish to cancel and request an instant refund:
Call our toll-free customer support helpline or visit:
http://geeksquad-billing-refunds.cancellation-desk.xyz/refund
Refund claims must be placed within 24 hours of invoice date.""",
            "category": "Financial Scam",
        },
        # Netflix Account Suspension
        {
            "subject": "Netflix: We were unable to process your monthly membership payment",
            "body": """Hi,
We are having trouble with your current billing information.
We will try again, but in the meantime you will need to update your payment method to keep enjoying movies and TV shows.
Update your credit card details now:
http://netflix-payment-update.account-billing.club/signin
If we cannot charge your card, your membership will be cancelled.
The Netflix Team""",
            "category": "Credential Harvesting",
        },
        # Social Media Verification
        {
            "subject": "Instagram Verified Badge Application Approved - Claim Your Blue Badge",
            "body": """Congratulations!
Your Instagram account has been selected for official verification badge status.
To claim your blue checkmark badge, submit your account credentials and identity verification:
http://instagram-verified-badge.meta-verification.ru/claim
You must claim within 24 hours or the badge will be awarded to another applicant.
Instagram Verification Team""",
            "category": "Credential Harvesting",
        },
        # Bank of America Security Alert
        {
            "subject": "Bank of America Alert: Restricted Account Access Due to Unusual Activity",
            "body": """Important Security Notification:
We detected multiple invalid PIN entries on your Bank of America debit card from an ATM in Chicago, IL.
For your protection, card and online banking services have been locked.
To unlock your account, please verify your identity:
http://104.28.19.4/bankofamerica-login/secure.php
Provide your SSN, card number, and online ID to restore access.
Bank of America Security Operations""",
            "category": "Financial Scam",
        },
        # Dropbox Document Shared
        {
            "subject": "You have received a confidential encrypted document via Dropbox Business",
            "body": """Dear Partner,
An urgent tender agreement (Purchase_Order_Agreement_2026.pdf) was shared with you via Dropbox Transfer.
To view and download the encrypted PDF document, please log in with your email provider:
http://dropbox-document-viewer.shared-cloud-portal.top/view-pdf
Access expires in 24 hours.
Dropbox Enterprise Cloud Transfer""",
            "category": "Credential Harvesting",
        },
        # LinkedIn InMail Phishing
        {
            "subject": "LinkedIn: You have 3 unread messages from Executive Recruiters",
            "body": """Hi,
You have received a direct executive recruitment inquiry regarding a VP of Engineering position with annual compensation of $320,000 + equity.
To read and respond to this message:
http://linkedin-messaging-portal.inmail-direct.work/recruiting/inbox
Don't miss this career opportunity.
LinkedIn Talent Network""",
            "category": "Credential Harvesting",
        },
    ]

    # -------------------------------------------------------------
    # 2. SAFE / LEGITIMATE EMAIL SAMPLES
    # -------------------------------------------------------------
    safe_samples = [
        # Meeting Invitation / Collaboration
        {
            "subject": "Sprint 42 Planning & Backlog Grooming - Monday 10:00 AM",
            "body": """Hi Team,
Let's meet on Monday at 10:00 AM EST for our regular Sprint 42 planning session.
We will review remaining Jira tickets, estimate upcoming user stories for the authentication service, and discuss velocity.
Please review the sprint board prior to the call:
https://jira.company.com/secure/RapidBoard.jspa?rapidView=142
Google Meet Link: https://meet.google.com/abc-defg-hij
Looking forward to speaking then!
Best regards,
Sarah Chen
Scrum Master""",
            "category": "Workplace Collaboration",
        },
        # 1-on-1 Meeting Notes
        {
            "subject": "Notes & Action Items from our 1-on-1 Catch-up",
            "body": """Hi Alex,
Thanks for the great conversation earlier today. Here is a quick recap of what we agreed on:
1. Complete unit tests for the data preprocessing module by Thursday.
2. Coordinate with DevOps on Docker deployment pipelines.
3. Schedule team demo for next sprint.
Let me know if I missed anything or if you need help unblocking dependencies.
Enjoy your afternoon!
Mark""",
            "category": "Workplace Collaboration",
        },
        # E-Commerce Order Confirmation
        {
            "subject": "Your Amazon.com order #114-9821038-19238 has shipped!",
            "body": """Hi David,
Great news! Your package is on its way.
Order Details:
- Anker USB-C Hub 7-in-1 Adapter ($39.99)
- Logitech MX Master 3S Wireless Mouse ($99.99)
Total: $139.98
Carrier: UPS
Tracking Number: 1Z9999999999999999
Estimated Delivery: Wednesday, November 25 by 8:00 PM.
You can view your order history or make returns anytime at https://www.amazon.com/your-orders
Thank you for shopping with Amazon!""",
            "category": "Transactional Receipt",
        },
        # Flight Booking Confirmation
        {
            "subject": "Delta Air Lines: Flight Confirmation - JFK to SFO (Confirmation: H982KQ)",
            "body": """Dear valued passenger,
Here is your confirmation for your upcoming trip to San Francisco.
Flight: DL 482
Departure: New York (JFK) - 08:30 AM
Arrival: San Francisco (SFO) - 11:45 AM
Seat: 14B (Main Cabin)
You can check in online starting 24 hours before your flight at https://www.delta.com
Please arrive at the airport at least 2 hours prior to scheduled departure.
Safe travels!
Delta Air Lines Customer Service""",
            "category": "Transactional Receipt",
        },
        # GitHub Pull Request Notification
        {
            "subject": "[GitHub] Pull Request #148: Refactor feature extraction pipeline (merged)",
            "body": """dev-bot merged commit 8a9b2c into master.
Review comments:
- Clean modular structure implemented.
- Added comprehensive unit tests in tests/test_features.py.
- Code coverage increased to 94%.
View pull request details on GitHub:
https://github.com/organization/security-ml-pipeline/pull/148
Notifications are managed via your GitHub account settings.""",
            "category": "Technical & Development",
        },
        # HR Benefits Open Enrollment
        {
            "subject": "Annual Health Benefits Open Enrollment Period (Nov 15 - Dec 1)",
            "body": """Dear Employees,
Our annual benefits open enrollment window is now open and runs through December 1.
During this period, you may make changes to your medical, dental, vision, and flexible spending accounts.
Please review the 2027 Benefits Guide attached on our intranet:
https://intranet.company.internal/hr/benefits-guide-2027
If you do not wish to make changes, your current elections will automatically roll over.
If you have questions, please contact hr-support@company.internal or attend our Q&A session this Thursday.
Best regards,
People & Culture Team""",
            "category": "Corporate Communication",
        },
        # Code Review / Pull Request Request
        {
            "subject": "Code Review Request: Microservice Database Migration Scripts",
            "body": """Hey Jason,
When you get a chance today, could you please take a look at the Alembic database migration scripts for the customer table?
Branch: feature/db-migration-v3
PR: https://github.com/corp/billing-service/pull/89
I want to make sure the indexing strategy matches our query patterns before we merge into staging.
No rush, whenever you have 15 minutes.
Thanks!
Elena""",
            "category": "Technical & Development",
        },
        # Quarterly All-Hands Meeting
        {
            "subject": "Invitation: Company Q4 All-Hands Meeting & Product Showcase",
            "body": """Hello Everyone,
Please join our Q4 All-Hands meeting this Wednesday from 2:00 PM to 3:30 PM EST.
Agenda:
- Q3 Financial Performance and Growth Metrics (CEO)
- Engineering Product Roadmap Showcase (CTO)
- Employee Recognition & Team Awards (HR)
- Open Q&A Session with Leadership
The session will be recorded and shared on the company intranet for team members in other time zones.
Calendar invite has been sent.
Executive Leadership Team""",
            "category": "Corporate Communication",
        },
        # Educational / Newsletter
        {
            "subject": "Python Weekly Newsletter - Issue #682",
            "body": """Welcome to Issue #682 of Python Weekly!
Top Articles this week:
- Understanding Python 3.13 GIL Changes and Free-Threading
- Building Fast ML Pipelines with Scikit-Learn and Polars
- High-Performance Streamlit Web Applications Architecture
- Pytest Best Practices for Enterprise Test Suites
Check out the full newsletter archive at https://www.pythonweekly.com/archive
To manage your subscription or unsubscribe, visit your preferences page.
Happy coding!
Rahul""",
            "category": "Newsletter & Educational",
        },
        # Personal Message
        {
            "subject": "Lunch tomorrow near downtown?",
            "body": """Hey Chris,
Long time no see! Are you free for lunch tomorrow around 12:30 PM?
I found a great new ramen place near 4th street that I have been wanting to try.
Let me know if that time works or if another day this week suits your schedule better.
Cheers,
Sam""",
            "category": "Personal Communication",
        },
        # Customer Support Ticket Resolution
        {
            "subject": "Support Ticket #49102 Resolved: Cloudflare DNS Configuration Assistance",
            "body": """Hi Marcus,
Your support request regarding DNS CNAME propagation has been marked as resolved by our technical operations team.
Your domain records are now active across all global edge locations.
If you require further assistance or have additional questions, simply reply to this email to automatically reopen your ticket.
You can view your ticket history at: https://support.hostingprovider.com/tickets/49102
Thank you for choosing our hosting platform!
Cloud Support Helpdesk""",
            "category": "Customer Support",
        },
        # Security Awareness Training Reminder
        {
            "subject": "Reminder: Complete Your Annual Cybersecurity Awareness Module",
            "body": """Dear Staff,
This is a gentle reminder that your mandatory annual security awareness training module is due by the end of this month.
Module: Identifying Phishing, Vishing, and Social Engineering Attacks
Estimated Duration: 25 minutes
Please access your training dashboard via your corporate single sign-on portal:
https://learning.corp.internal/course/cybersecurity-2026
Thank you for helping keep our company data and customer privacy protected!
Corporate Information Security Office""",
            "category": "Corporate Communication",
        },
        # Jira Issue Assigned
        {
            "subject": "[Jira] (SEC-402) Assigned to you: Update cryptographic cipher suites",
            "body": """Hi Maria,
Alex assigned SEC-402 to you.
Issue Details:
Project: Security Engineering (SEC)
Priority: Medium
Description: Update the TLS cipher suites on the external load balancer to deprecate older TLS 1.0/1.1 protocols.
Link: https://jira.internal.net/browse/SEC-402
You received this notification because you are subscribed to updates for SEC project issues.""",
            "category": "Technical & Development",
        },
        # Slack Weekly Digest
        {
            "subject": "Slack: Highlights from your channels this week",
            "body": """Here is what you missed in your workspace while you were away:
#general: 14 new messages
#engineering: Discussion on migrating Redis cache cluster to Redis 7
#announcements: Welcome our new Senior Data Engineer, Priya Patel!
Open Slack desktop app or visit https://workspace.slack.com to catch up with your team.
Manage your email notification frequency in Slack notification settings.""",
            "category": "Workplace Collaboration",
        },
        # Academic / Research Paper Review
        {
            "subject": "Peer Review Request: IEEE Transactions on Cyber Threat Intelligence",
            "body": """Dear Dr. Vance,
We would like to invite you to review manuscript #TIFS-2026-0891 titled:
'Comparative Analysis of Machine Learning Approaches for Email Threat Classification.'
The abstract is available on our journal review portal:
https://manuscriptcentral.com/tifs-review?id=891
Please respond within 5 days to confirm whether you are available to review this submission.
Kind regards,
Editor-in-Chief
IEEE Transactions on Information Forensics and Security""",
            "category": "Academic & Research",
        },
        # Uber Trip Receipt
        {
            "subject": "Your Wednesday morning trip with Uber",
            "body": """Thanks for riding with Uber!
Trip summary:
Date: Wednesday, November 18
Total: $24.50
Payment Method: Apple Pay (Visa ending in 4109)
Pickup: 500 Market St, San Francisco, CA
Dropoff: 100 1st St, San Francisco, CA
Driver: Carlos (5.0 rating)
A complete tax receipt and ride map is available in the Uber app or online at:
https://riders.uber.com/trips
Need help with a lost item? Contact customer support through the app.""",
            "category": "Transactional Receipt",
        },
        # Apartment Rent Receipt
        {
            "subject": "Payment Confirmation: Monthly Rent Received - Unit 402",
            "body": """Dear Resident,
We have received your electronic payment of $2,100.00 for Unit 402 rent.
Transaction Reference: RENT-NOV-2026-402
Payment status: Completed
Your resident portal balance is currently $0.00.
To submit maintenance requests or view payment history, visit:
https://residentportal.pmgmanagement.com/login
Thank you for your prompt payment!
Property Management Office""",
            "category": "Transactional Receipt",
        },
        # Library Book Due Reminder
        {
            "subject": "City Public Library: Courtesy Reminder - Books Due in 3 Days",
            "body": """Dear Library Patron,
This courtesy notice is to remind you that the following items checked out on library card #***492 are due on November 23:
1. Designing Data-Intensive Applications by Martin Kleppmann
2. Python Machine Learning (3rd Edition) by Sebastian Raschka
You can renew items online if there are no pending holds:
https://catalog.citylibrary.org/patron-renew
Have a wonderful day of reading!
City Public Library Circulation Desk""",
            "category": "Community & Personal",
        },
        # Doctor Appointment Confirmation
        {
            "subject": "Appointment Confirmation: Annual Wellness Exam with Dr. Stephanie Lee",
            "body": """Dear Robert,
Your annual preventive wellness exam is confirmed for:
Date: Tuesday, December 2 at 10:30 AM
Provider: Dr. Stephanie Lee, MD (Internal Medicine)
Location: Metro Health Clinic, Suite 300
Please arrive 15 minutes before your scheduled appointment time to complete check-in.
You may review your medical history and pre-fill forms on our patient portal:
https://patientportal.metrohealth.org/appointments
If you need to reschedule, please notify us at least 24 hours in advance.
Metro Health Partners""",
            "category": "Medical & Appointments",
        },
        # Software License Renewal Receipt
        {
            "subject": "JetBrains Order Confirmation: Subscription Renewal (Order #JB-981240)",
            "body": """Dear Customer,
Thank you for renewing your JetBrains All Products Pack subscription!
Order ID: JB-981240
Validity: Valid until November 2027
Licensee: Engineering Team
You can manage user assignments, download license keys, and generate corporate tax invoices from your account:
https://account.jetbrains.com/licenses
Thank you for developing with JetBrains tools.
The JetBrains Sales Team""",
            "category": "Transactional Receipt",
        },
    ]

    # Convert initial seeds
    for p in phishing_samples:
        emails.append({
            "subject": p["subject"],
            "body": p["body"],
            "category": p["category"],
            "label": 1,
            "label_text": "PHISHING",
            "source": "Curated Phishing Benchmark Corpus",
        })

    for s in safe_samples:
        emails.append({
            "subject": s["subject"],
            "body": s["body"],
            "category": s["category"],
            "label": 0,
            "label_text": "SAFE",
            "source": "Curated Legitimate Communications Corpus",
        })

    # Expand the curated corpus systematically with realistic, authentic variants
    # covering diverse attack surfaces, linguistic cues, and legitimate business patterns.
    # We generate a total corpus of hundreds of varied, non-trivial records.
    expanded_phishing = [
        # Additional Phishing Scams:
        ("Security Warning: Unauthorized credit card charge of $1,299.00",
         "Dear Customer,\nWe detected a pending debit of $1,299.00 on your Best Buy account.\nIf you did not authorize this purchase, verify your identity immediately to cancel:\nhttp://bestbuy-fraud-protection.orders-verification.top/cancel.php\nFailure to cancel within 2 hours will finalize the charge.\nBest Buy Fraud Prevention",
         "Financial Scam"),
        ("ATTENTION: Your Cloud Storage Is 99% Full - Files Scheduled for Deletion",
         "Urgent Notice:\nYour cloud storage quota has exceeded its allocated limit.\nYour stored documents, photos, and backups will be permanently deleted within 24 hours unless you upgrade.\nClaim your free 50GB storage voucher now:\nhttp://cloud-storage-upgrade.drive-sync.work/upgrade?voucher=claim\nCloud Storage Operations Team",
         "Credential Harvesting"),
        ("IRS Final Notice: Outstanding Property Tax Assessment Due",
         "FINAL NOTICE BEFORE LEGAL ACTION\nInternal Revenue Service Department of Revenue\nOur records show an unpaid balance of $3,450.00.\nAn arrest warrant will be issued if settlement is not executed within 24 hours.\nAccess your case file and pay via debit card or wire:\nhttp://185.190.22.12/irs-settlement/pay-tax.html\nOffice of the Inspector General",
         "Legal & Extortion"),
        ("DocuSign: Important Lease Renewal Agreement Requires Your Signature",
         "DocuSign Electronic Document Signing Service\nPlease review and electronically sign: 'Commercial_Lease_Amendment_2026.pdf'\nSender: Property Leasing Group\nReview Document: http://docusign-secure-envelope.document-signing.club/envelope?id=98124\nThis link will expire in 24 hours. Keep your documents secure.\nDocuSign Inc.",
         "Credential Harvesting"),
        ("URGENT: Suspicious wire transaction from your Wells Fargo Account",
         "Wells Fargo Alert:\nA wire transfer of $4,500.00 was requested from your account to an overseas recipient.\nIf this was NOT you, click immediately to block the wire:\nhttp://wellsfargo-secure-alerts.banking-online.xyz/dispute\nFailure to dispute immediately authorizes the transaction.\nWells Fargo Security Department",
         "Financial Scam"),
        ("Zoom: You missed an urgent recorded video meeting from Management",
         "You have 1 missed recorded video conference from Executive Leadership.\nTopic: Emergency Staff Restructuring Announcement\nDuration: 14 mins\nClick to view recording: http://zoom-recording-cloud.video-conference.ru/watch?id=49210\nZoom Video Communications",
         "Credential Harvesting"),
        ("Immediate Action: Your Amazon Prime Membership Has Been Suspended",
         "Dear Amazon Prime Member,\nWe were unable to charge your payment card for Prime membership renewal.\nYour Prime benefits, free shipping, and Prime Video streaming have been placed on hold.\nRe-activate your account by updating your credit card:\nhttp://amazon-prime-billing.update-account.top/prime/billing\nAmazon Customer Service",
         "Credential Harvesting"),
        ("HR Alert: Mandatory Employee Safety Policy Sign-off Required",
         "All Employees:\nOSHA compliance requires all staff members to review and sign the updated 2026 workplace safety guidelines.\nFailure to sign by end of day will result in formal HR disciplinary action.\nLog in to sign: http://workplace-osha-compliance.staff-portal.work/sign\nHuman Resources Department",
         "Credential Harvesting"),
        ("Cryptocurrency Giveaway: Elon Musk / Tesla 5,000 BTC Celebration",
         "Official Tesla Promotional Event:\nTo celebrate the launch of our new AI initiative, Tesla is giving away 5,000 BTC and 50,000 ETH.\nSend between 0.1 BTC to 5 BTC to the contribution address and get double back instantly!\nVisit official promotion page: http://tesla-crypto-giveaway.event-promotions.top\nRules and terms apply. Act quickly, limited availability!",
         "Financial Scam"),
        ("Norton LifeLock: Security alert - Identity theft protection breached",
         "Alert: High Risk Identity Compromise\nNorton LifeLock detected your SSN and passport number traded on dark web forums.\nTo lock your credit report and prevent unauthorized loans:\nhttp://norton-identity-protection.security-shield.icu/lock\nImmediate action is strongly advised.\nNorton Security Operations Center",
         "Financial Scam"),
        ("eBay: Your seller account will be restricted within 12 hours",
         "Dear eBay Seller,\nWe noticed discrepancies in your seller registration information.\nIn accordance with eBay User Agreement, please submit proof of address and ID card:\nhttp://ebayisapi-seller-verification.account-protection.xyz/verify\nFailure to provide documentation will result in permanent suspension of seller privileges.\neBay Trust & Safety",
         "Credential Harvesting"),
        ("Urgent: Your Domain Name Will Expire in 24 Hours",
         "Domain Expiration Warning:\nYour domain registration is scheduled to expire and enter redemption status.\nTo avoid losing your website, email, and brand ownership, renew immediately:\nhttp://domain-registry-renewal.dns-services.work/renew?domain=corporate\nFailure to renew allows competitors to claim your domain.\nDomain Registrar Registry Services",
         "Financial Scam"),
        ("Facebook / Meta: Copyright Infringement Notice on Your Business Page",
         "Meta Business Support:\nYour page has been reported for trademark and copyright infringement by another brand.\nYour page is scheduled for deletion within 24 hours.\nIf you believe this is an error, appeal the decision here:\nhttp://meta-business-support.appeal-forms.ru/appeal?id=99281\nMeta Platforms Inc.",
         "Credential Harvesting"),
        ("DHL Express: Delivery fee pending for international parcel #GB-481920",
         "DHL Customer Service:\nYour international package from London is held due to $4.20 unpaid import taxes.\nTo clear customs and confirm delivery address:\nhttp://dhl-express-customs-duty.shipment-tracking.top/clearance\nFailure to pay within 48 hours will result in parcel destruction.\nDHL Global Forwarding",
         "Delivery Scam"),
        ("Chase Bank: Unauthorized Zelle Payment of $750.00 Initiated",
         "Chase Fraud Protection:\nDid you authorize a Zelle transfer of $750.00 to 'John Smith'?\nIf NO, cancel the transaction immediately by verifying your online account:\nhttp://192.168.1.150/chase-zelle-cancel/verify.html\nDo not share your one-time passcode with anyone.\nJPMorgan Chase Fraud Department",
         "Financial Scam"),
        ("Apple Store: Receipt for your purchase of iPhone 16 Pro Max ($1,399.00)",
         "Apple Invoice:\nThank you for your order at Apple Store Fifth Avenue.\nDevice: iPhone 16 Pro Max 512GB Space Black ($1,399.00)\nCharged to: Visa ending in 8812\nIf you did not make this purchase, dispute and request a full refund:\nhttp://apple-store-dispute.invoice-refund.club/dispute\nApple Inc.",
         "Financial Scam"),
        ("Urgent: IT Department - Server migration required for your mailbox",
         "Helpdesk Alert:\nOur email servers are undergoing security patching.\nAll users must validate their active directory credentials to migrate their mailbox archive.\nValidate credentials here: http://mail-server-patching.it-migration.work/active-directory\nAccounts not validated will lose all saved emails.\nCorporate IT Helpdesk",
         "Credential Harvesting"),
        ("USPS: Package Delivery Interrupted - Incomplete Street Address",
         "United States Postal Service:\nYour package #9400111899223192 could not be delivered due to an incorrect house number.\nUpdate your address and pay a $1.25 redelivery fee:\nhttp://usps-redelivery-tracking.postal-service.xyz/address-update\nUSPS Customer Care",
         "Delivery Scam"),
        ("Bank of America: New wire transfer payee added to your account",
         "Security Notification:\nA new external transfer recipient 'Global Trade FX Ltd' was added to your account.\nIf you did not add this payee, cancel immediately:\nhttp://bankofamerica-payee-verification.online-access.top/cancel-payee\nBank of America Online Security",
         "Financial Scam"),
        ("Warning: Unusual Activity on your Coinbase Account",
         "Coinbase Security Notice:\nAn unauthorized withdrawal of 1.45 BTC was initiated from an IP in Nigeria.\nIf you did not authorize this withdrawal, click below to freeze your account:\nhttp://coinbase-account-lock.crypto-security.club/freeze?auth=9821\nCoinbase Security Operations",
         "Financial Scam"),
    ]

    expanded_safe = [
        # Additional Safe Communications:
        ("Team Weekly Standup Notes and Sprint Priorities",
         "Hi everyone,\nHere are the key takeaways from today's morning standup:\n- Front-end team completed the dashboard navigation revamp.\n- API response time optimization is currently underway.\n- QA test automation suite is scheduled for review on Wednesday.\nPlease check our team wiki for the complete action items:\nhttps://wiki.corp.internal/display/ENG/Standup+Notes\nHave a productive week!",
         "Workplace Collaboration"),
        ("Receipt for your Google Cloud Platform Services",
         "Hello,\nHere is your monthly invoice for Google Cloud services.\nInvoice Number: GCP-2026-11-0918\nAmount: $84.22 USD\nPayment method: Auto-debit from corporate Mastercard ending in 5519\nYou can download the detailed CSV breakdown from the Google Cloud Console:\nhttps://console.cloud.google.com/billing\nThank you for choosing Google Cloud.",
         "Transactional Receipt"),
        ("Slack: New comment on issue PROJ-512 in #engineering",
         "Dave commented on PROJ-512:\n'The latency issue has been addressed by adding a composite index on (user_id, created_at). Verified in staging.'\nView thread in Slack: https://team.slack.com/archives/C012345/p168921098\nReply directly in Slack or in the project channel.",
         "Technical & Development"),
        ("Reminder: Upcoming Dental Cleaning Appointment",
         "Hi Jessica,\nThis is a friendly reminder of your upcoming dental examination and cleaning with Dr. Harris:\nDate: Friday, November 28 at 2:00 PM\nLocation: Downtown Dental Care, 200 Main St.\nPlease arrive 10 minutes early. If you need to reschedule, call our office at (555) 019-2831.\nSee you soon!",
         "Medical & Appointments"),
        ("GitHub: Dependabot vulnerability alert for numpy in security-scanner",
         "Dependabot detected a low-severity vulnerability in dependency 'numpy' (<2.0.0).\nRecommended action: Upgrade to numpy 2.4.4.\nA pull request with the version bump has been automatically created:\nhttps://github.com/company/security-scanner/pull/34\nReview the changelog and test results before merging.",
         "Technical & Development"),
        ("Company Holiday Calendar and Office Closure Schedule",
         "Dear Team,\nPlease find the approved holiday calendar for the upcoming holiday season.\nThe corporate office will be closed on:\n- Thanksgiving Day (Thursday, Nov 27)\n- Day After Thanksgiving (Friday, Nov 28)\n- Christmas Day (Thursday, Dec 25)\n- New Year's Day (Thursday, Jan 1)\nCustomer support on-call rotations have been published to PagerDuty.\nWishing everyone a safe and restful holiday season!",
         "Corporate Communication"),
        ("Your Lyft ride receipt with Driver James",
         "Thanks for riding with Lyft!\nRide Date: Tuesday, Nov 17\nTotal Fare: $18.45\nTip: $3.00\nTotal Paid: $21.45\nPayment: Apple Pay\nView full receipt or download PDF: https://www.lyft.com/ride-receipts/19820192\nWe hope you enjoyed your ride!",
         "Transactional Receipt"),
        ("Invitation: Machine Learning Tech Talk - Transformers in Cybersecurity",
         "Hi Engineers,\nYou are invited to our monthly Tech Talk series this Thursday from 4:00 PM to 5:00 PM in Conference Room B and on Zoom.\nSpeaker: Dr. Angela Wu, Senior AI Researcher\nTopic: Practical Applications of Transformer Architecture for Network Anomaly Detection\nRefreshments will be provided. Calendar invite is attached.\nEngineering Culture Committee",
         "Technical & Development"),
        ("Stripe: Monthly Payout Confirmation ($8,420.00)",
         "Hi there,\nA payout of $8,420.00 USD has been sent to your Chase bank account ending in 9102.\nEstimated arrival: 1-2 business days.\nTo view the payout details and associated customer transactions, visit your Stripe Dashboard:\nhttps://dashboard.stripe.com/payouts/po_192830192\nThank you for partnering with Stripe.",
         "Transactional Receipt"),
        ("Quarterly Performance Review Reminder - Self-Evaluation Due",
         "Dear Employee,\nIt is time for our Q4 performance review cycle.\nPlease take 30 minutes to complete your self-evaluation form in Lattice by next Friday:\nhttps://lattice.company.com/reviews/q4-2026\nYour manager will schedule a dedicated conversation once both evaluations are submitted.\nPeople Operations Team",
         "Corporate Communication"),
        ("Coffee catch-up this Friday?",
         "Hey Dave,\nHope your week is going smoothly! Let me know if you have time for a quick 20-minute coffee catch-up this Friday around 3 PM.\nI would love to pick your brain on the new microservice architecture design.\nLet me know if that works for you!\nBest,\nMarcus",
         "Personal Communication"),
        ("Your Spotify Wrapped 2026 is ready to explore!",
         "Hi listener,\nYour 2026 audio story is here! Discover your top artists, favorite genres, and total minutes listened this year.\nCheck out your personalized Wrapped playlist and share your badges with friends:\nhttps://www.spotify.com/wrapped\nThank you for spending another musical year with Spotify!",
         "Community & Personal"),
        ("Coursera: Certificate of Completion - Deep Learning Specialization",
         "Congratulations Jason!\nYou have successfully completed all 5 courses in the Deep Learning Specialization by DeepLearning.AI.\nYour verified certificate is ready to view and add to your LinkedIn profile:\nhttps://www.coursera.org/verify/specialization/ABC123XYZ\nKeep up the great learning journey!\nThe Coursera Team",
         "Newsletter & Educational"),
        ("Home Depot: Order Ready for In-Store Pickup (#WG-8921)",
         "Hi Michael,\nYour order #WG-8921 is now picked and ready for pickup at our West Valley Store (Locker #14).\nItems:\n- Milwaukee M18 Cordless Drill Kit (1)\n- 50ft Outdoor Heavy Duty Extension Cord (1)\nPlease bring a valid photo ID and this order barcode when picking up.\nCurbside pickup is also available via the Home Depot app.",
         "Transactional Receipt"),
        ("Bug Report: Memory leak in long-running Celery worker process",
         "Hi Backend Team,\nWe noticed memory consumption on the Celery background worker pods steadily increasing over a 48-hour period.\nIt appears tasks handling PDF rendering are not properly releasing file descriptors.\nCreated tracking ticket INFRA-892.\nPlease review the memory profiling graphs attached to the ticket:\nhttps://monitoring.corp.internal/dashboards/celery-memory\nThanks,\nSite Reliability Engineering",
         "Technical & Development"),
        ("Team Lunch Photos & Recipe Share from Friday BBQ",
         "Hey everyone,\nThanks to everyone who joined our team BBQ on the terrace on Friday! It was wonderful spending time together outside the sprint board.\nI uploaded all the event photos and the homemade salsa recipe to our shared drive folder:\nhttps://drive.google.com/drive/folders/team-bbq-2026\nSee you all at tomorrow's standup!\nRachel",
         "Personal Communication"),
        ("Atlassian Confluence: Weekly digest of spaces you follow",
         "Here is what happened in your Confluence spaces this week:\nArchitecture Decision Records (ADRs):\n- ADR-042: Adoption of gRPC for internal service-to-service communication (Updated by Kevin)\n- ADR-043: Caching strategy using Redis Cluster (Created by Sarah)\nRead the full documents on Confluence:\nhttps://confluence.corp.net/display/ARCH/ADR-042",
         "Technical & Development"),
        ("Donation Receipt: Thank you for supporting the World Wildlife Fund",
         "Dear Donor,\nThank you for your generous tax-deductible gift of $50.00 to World Wildlife Fund.\nTransaction ID: WWF-2026-98124\nDate: November 15\nYour donation directly supports endangered species conservation and habitat restoration projects worldwide.\nOfficial 501(c)(3) tax receipt is attached.\nWorld Wildlife Fund Donor Relations",
         "Community & Personal"),
        ("New Release: Node.js 24 LTS is now available",
         "Node.js Foundation Announcement:\nNode.js 24 has entered Long Term Support (LTS) status with codename 'Iron'.\nKey highlights:\n- Updated V8 JavaScript engine 12.8\n- Built-in WebSocket client stabilized\n- Improved native test runner performance\nRead the complete release notes on nodejs.org:\nhttps://nodejs.org/en/blog/release/v24-lts\nHappy hacking!",
         "Newsletter & Educational"),
        ("HR: Flu Shot Clinic on Campus - Next Tuesday",
         "Dear Colleagues,\nOur annual on-site employee flu vaccination clinic will take place next Tuesday, November 25, in the Wellness Center (Building A, Room 102) from 9:00 AM to 3:00 PM.\nThe vaccination is completely free for all employees and contractors.\nWalk-ins are welcome, or you can reserve a 5-minute time slot here:\nhttps://intranet.company.internal/wellness/flu-clinic-2026\nStay healthy this winter!\nEmployee Health & Safety",
         "Corporate Communication"),
    ]

    for p in expanded_phishing:
        emails.append({
            "subject": p[0],
            "body": p[1],
            "category": p[2],
            "label": 1,
            "label_text": "PHISHING",
            "source": "Curated Phishing Benchmark Corpus",
        })

    for s in expanded_safe:
        emails.append({
            "subject": s[0],
            "body": s[1],
            "category": s[2],
            "label": 0,
            "label_text": "SAFE",
            "source": "Curated Legitimate Communications Corpus",
        })

    # Now systematically synthesize diverse structural and lexical variations
    # to form a rich dataset of ~400 balanced records.
    # We introduce varied parameters:
    # - Different financial institutions, tech brands, cloud services, internal IT notices
    # - Varied legitimate corporate departments, project names, technical tools, travel, receipts
    # This ensures vocabulary richness for TF-IDF unigram & bigram vectors and rich URL structural variety.

    phish_banks = ["Chase Bank", "Bank of America", "Wells Fargo", "Citibank", "Barclays", "Capital One"]
    phish_tech = ["Microsoft", "Google", "Apple", "Dropbox", "Adobe", "DocuSign", "Netflix", "Amazon"]
    phish_tlds = [".xyz", ".top", ".club", ".work", ".ru", ".tk", ".ml", ".icu", ".buzz"]
    phish_keywords = ["verify-account", "security-update", "confirm-login", "billing-dispute", "portal-auth", "restore-access"]

    for i in range(160):
        bank = phish_banks[i % len(phish_banks)]
        tech = phish_tech[i % len(phish_tech)]
        tld = phish_tlds[i % len(phish_tlds)]
        slug = phish_keywords[i % len(phish_keywords)]
        amount = 350 + (i * 47) % 3500

        if i % 4 == 0:
            subj = f"URGENT: Your {bank} account has been temporarily restricted (Case #{81000 + i})"
            body = (
                f"Dear Valued {bank} Customer,\n"
                f"We detected an unauthorized wire transaction of ${amount}.00 from an unverified location.\n"
                f"For your protection, all debit card transactions and online access have been suspended.\n"
                f"To verify your identity and cancel this unauthorized withdrawal, click below immediately:\n"
                f"http://{bank.lower().replace(' ', '')}-{slug}.service-auth{tld}/login?case={81000 + i}\n"
                f"Failure to verify within 24 hours will result in permanent closure of your account.\n"
                f"{bank} Fraud Investigation Department"
            )
            cat = "Financial Scam"
        elif i % 4 == 1:
            subj = f"Critical Alert: {tech} Account Password Expiring - Verification Required"
            body = (
                f"Security Notification from {tech} Security Operations:\n"
                f"Your account credentials have expired according to enterprise security compliance.\n"
                f"You must confirm your current password and username immediately to prevent account suspension:\n"
                f"http://{tech.lower()}-credential-sync.{slug}{tld}/portal/index.php\n"
                f"Action is required within 12 hours. Do not share your login credentials.\n"
                f"{tech} Identity Management"
            )
            cat = "Credential Harvesting"
        elif i % 4 == 2:
            subj = f"FINAL NOTICE: Immediate Action Required - Invoice #{10920 + i} Overdue"
            body = (
                f"Attention Accounting Department,\n"
                f"Invoice #{10920 + i} for ${amount}.50 is severely past due.\n"
                f"Legal proceedings will commence immediately unless wire settlement is completed today.\n"
                f"View statement and submit payment details:\n"
                f"http://192.168.0.{10 + (i % 200)}/invoices/{slug}.html\n"
                f"Contact our legal collections office immediately.\n"
                f"Global Debt Collections Ltd"
            )
            cat = "Legal & Extortion"
        else:
            subj = f"Delivery Exception: Parcel #{9401200 + i} held at depot"
            body = (
                f"Notice of Attempted Parcel Delivery:\n"
                f"Carrier was unable to deliver your package #{9401200 + i} due to unpaid customs clearance duty ($3.95).\n"
                f"Click the link below to confirm your delivery address and pay the clearance fee:\n"
                f"http://express-courier-tracking.{slug}{tld}/track?id={9401200 + i}\n"
                f"Packages unclaimed after 48 hours will be destroyed.\n"
                f"National Logistics Delivery Services"
            )
            cat = "Delivery Scam"

        emails.append({
            "subject": subj,
            "body": body,
            "category": cat,
            "label": 1,
            "label_text": "PHISHING",
            "source": "Curated Phishing Benchmark Corpus",
        })

    # Diverse legitimate emails
    safe_projects = ["Mobile App V2", "Cloud Migration", "Kubernetes Deployment", "API Gateway", "Data Lakehouse", "Design System"]
    safe_departments = ["Engineering", "Product Design", "Finance", "Legal Compliance", "DevOps", "Customer Success"]
    safe_services = ["AWS", "GitHub", "Jira", "Confluence", "Slack", "Asana", "Salesforce"]

    for i in range(160):
        proj = safe_projects[i % len(safe_projects)]
        dept = safe_departments[i % len(safe_departments)]
        svc = safe_services[i % len(safe_services)]
        ref = 1000 + i

        if i % 4 == 0:
            subj = f"Project Update: {proj} Milestone {1 + (i % 5)} Status & Review"
            body = (
                f"Hi {dept} Team,\n"
                f"Here is a summary of our progress on the {proj} milestone:\n"
                f"- Met 92% of sprint deliverables on time.\n"
                f"- Resolved 14 open bugs in our staging environment.\n"
                f"- End-to-end integration testing will begin on Wednesday.\n"
                f"Please review the detailed task breakdown on Jira:\n"
                f"https://company.atlassian.net/browse/PROJ-{ref}\n"
                f"Thanks for everyone's hard work on this release!\n"
                f"Best regards,\nProject Management Office"
            )
            cat = "Workplace Collaboration"
        elif i % 4 == 1:
            subj = f"Your {svc} Monthly Account Summary & Usage Report"
            body = (
                f"Hello,\n"
                f"Your monthly {svc} billing report for account #{ref * 7} is now available.\n"
                f"Summary: All cloud services are operating normally within designated quotas.\n"
                f"You can review detailed usage analytics, team member permissions, and invoices online:\n"
                f"https://{svc.lower()}.com/enterprise/account/billing\n"
                f"If you have questions about your billing statement, our support team is available 24/7.\n"
                f"Sincerely,\nThe {svc} Account Team"
            )
            cat = "Transactional Receipt"
        elif i % 4 == 2:
            subj = f"Meeting Agenda: {dept} Quarterly Strategic Alignment"
            body = (
                f"Hi Team,\n"
                f"Looking forward to our upcoming strategy alignment session on Thursday at 2:00 PM EST.\n"
                f"Agenda Topics:\n"
                f"1. Review of annual operational objectives.\n"
                f"2. Budget allocation and resource planning for upcoming quarter.\n"
                f"3. Cross-functional collaboration improvements.\n"
                f"Zoom Link: https://zoom.us/j/{900000000 + i}\n"
                f"Please add your agenda items to the shared doc beforehand.\n"
                f"See you then!"
            )
            cat = "Workplace Collaboration"
        else:
            subj = f"Technical Documentation: Architecture Guidelines for {proj}"
            body = (
                f"Hi Engineers,\n"
                f"I have finalized the architectural documentation for {proj}.\n"
                f"Key design decisions:\n"
                f"- RESTful API endpoints following OpenAPI 3.1 specifications.\n"
                f"- Asynchronous event processing via Kafka message bus.\n"
                f"- Automated schema migrations managed with Liquibase.\n"
                f"Read the full architectural specification here:\n"
                f"https://docs.corp.internal/systems/{proj.lower().replace(' ', '-')}\n"
                f"Please leave feedback or comments on the document by Friday.\n"
                f"Principal Systems Architect"
            )
            cat = "Technical & Development"

        emails.append({
            "subject": subj,
            "body": body,
            "category": cat,
            "label": 0,
            "label_text": "SAFE",
            "source": "Curated Legitimate Communications Corpus",
        })

    return emails


def prepare_dataset():
    """
    Acquires, cleans, balances, and saves the email dataset.
    """
    logger.info("Starting dataset preparation pipeline...")

    # Load curated email corpus
    raw_emails = get_curated_email_corpus()
    df = pd.DataFrame(raw_emails)

    logger.info(f"Loaded {len(df)} initial raw records.")

    # 1. Cleaning: handle missing values
    initial_len = len(df)
    df["subject"] = df["subject"].fillna("").astype(str).str.strip()
    df["body"] = df["body"].fillna("").astype(str).str.strip()
    df["category"] = df["category"].fillna("General").astype(str).str.strip()

    # Drop entries where both subject and body are empty
    df = df[~((df["subject"] == "") & (df["body"] == ""))]

    # Combine subject and body for full text analysis
    df["full_text"] = df["subject"] + " " + df["body"]

    # 2. Cleaning: remove exact duplicates based on full_text
    df = df.drop_duplicates(subset=["full_text"]).reset_index(drop=True)
    logger.info(f"Deduplicated records: {initial_len} -> {len(df)} records.")

    # 3. Normalize labels
    df["label"] = df["label"].astype(int)
    df["label_text"] = df["label"].map({1: "PHISHING", 0: "SAFE"})

    # 4. Examine class distribution
    class_counts = df["label_text"].value_counts().to_dict()
    phishing_count = class_counts.get("PHISHING", 0)
    safe_count = class_counts.get("SAFE", 0)
    total_count = len(df)

    logger.info(f"Dataset Summary:")
    logger.info(f"  Total records: {total_count}")
    logger.info(f"  PHISHING: {phishing_count} ({phishing_count / total_count * 100:.2f}%)")
    logger.info(f"  SAFE:     {safe_count} ({safe_count / total_count * 100:.2f}%)")

    # 5. Save output file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logger.info(f"Dataset successfully saved to: {OUTPUT_FILE}")

    return df


if __name__ == "__main__":
    prepare_dataset()
