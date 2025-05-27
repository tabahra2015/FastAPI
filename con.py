import os
import subprocess

# المجلد الذي يحتوي على الملفات
folder_path = 'converted_wav'

# أنواع الملفات المدعومة للتحويل
input_extensions = ['.mp3', '.ogg', '.flac', '.m4a', '.wav']

for filename in os.listdir(folder_path):
    name, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext in input_extensions:
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(folder_path, f"{name}_converted.wav")

        # أمر ffmpeg للتحويل إلى WAV PCM 16-bit
        command = [
            'ffmpeg',
            '-y',                      # استبدال الملف إذا موجود
            '-i', input_path,          # الملف الأصلي
            '-acodec', 'pcm_s16le',    # الترميز: PCM 16-bit
            '-ar', '16000',            # sample rate: 16000 Hz
            '-ac', '1',                # mono
            output_path                # الملف الناتج
        ]

        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f'✅ Converted: {filename} → {name}_converted.wav')
