import pandas as pd
import os
from twilio.rest import Client

# Twilio credentials
with open("/cs/usr/alina.ryabtsev/twilio_credentials.txt", "r") as f:
    text = f.read().split("\n")
    account_sid = text[0].split(" ")[1]
    auth_token = text[1].split(" ")[1]
twilio_phone_number = '+14155238886'
your_phone_number = '+972544987857'

# Set up the Twilio client
client = Client(account_sid, auth_token)

# Your message content
message_body = "Burning DICOM has finished!"


# Read the Excel file
excel_file_path = '/cs/casmip/alina.ryabtsev/Tools/HCC cases.xlsx'
dest_folder_path = "/cs/casmip/alina.ryabtsev/PrimaryLesionsProject/Dataset/CT/HCC"
input_dir = '/media/alina.ryabtsev/PATIENT_DATA'
dicom_convert_cmd = '//cs/usr/alina.ryabtsev/dcm2niix/dcm2niix -z y -o'
df = pd.read_excel(excel_file_path)

# Iterate through each row in the DataFrame
while True:
    scan_id = input("Enter scan ID: ")

    # Validate input scan id
    if not scan_id.isnumeric():
        print("Scan ID is not valid. Try numeric ID")
        continue

    scan_row = df.loc[df['Scan IDs'] == int(scan_id)]

    if scan_row.empty:
        print("Scan ID not found. Try again with another ID.")
        continue

    # Extract name and date from th e current row
    patient = scan_row['Patient Case'].iloc[0].replace(" ", "")
    date = pd.to_datetime(scan_row["Scans"]).dt.strftime("%Y-%m-%d").iloc[0].replace(" ", "")

    # Create a folder with the patient if it doesn't exist
    folder_path = os.path.join(dest_folder_path, patient)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create a folder with the name from the 'Scans' column inside the created folder
    output_folder = os.path.join(folder_path, date)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print("Disk not found. Insert a disk or try again later.")
        continue

    # Check if the output folder is not empty
    if os.listdir(output_folder):
        print("DICOM is already converted. Try again with another ID.")
        continue

    # Burn DICOM into created sub folder
    print(f"\nConverting DICOM {scan_id} to NIFTI...\n")
    os.system(f"{dicom_convert_cmd} {output_folder} {input_dir}")
    print("\nDICOM converted successfully.\n")
    print('─' * 10)  # U+2501, Box Drawings Heavy Horizontal

    # Send the WhatsApp message
    message = client.messages.create(
        from_='whatsapp:{}'.format(twilio_phone_number),
        body=message_body,
        to='whatsapp:{}'.format(your_phone_number)
    )

