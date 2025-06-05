import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv('uji_pencahayaan_log.csv')

# Convert timestamp to readable format (optional)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Plot brightness over time
plt.figure(figsize=(10,4))
plt.plot(df['timestamp'], df['brightness'], label='Brightness')
plt.xlabel('Time')
plt.ylabel('Brightness')
plt.title('Brightness Over Time')
plt.legend()
plt.show()

# Plot FPS over time
plt.figure(figsize=(10,4))
plt.plot(df['timestamp'], df['fps'], label='FPS', color='orange')
plt.xlabel('Time')
plt.ylabel('FPS')
plt.title('FPS Over Time')
plt.legend()
plt.show()

# Plot number of people detected over time
plt.figure(figsize=(10,4))
plt.plot(df['timestamp'], df['detected_person_count'], label='Detected Persons', color='green')
plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Detected Persons Over Time')
plt.legend()
plt.show()
