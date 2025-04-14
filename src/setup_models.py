import os
import sys
import urllib.request
import bz2
import shutil

def download_file(url, filename):
    """Download a file from URL with progress indicator"""
    print(f"Downloading {filename}...")
    try:
        with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
            file_size = int(response.headers['Content-Length'])
            chunk_size = 8192
            downloaded = 0
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                downloaded += len(chunk)
                out_file.write(chunk)
                
                # Show progress
                progress = int(50 * downloaded / file_size)
                sys.stdout.write('\r[{}{}] {}%'.format(
                    '#' * progress,
                    '.' * (50 - progress),
                    int(100 * downloaded / file_size)
                ))
                sys.stdout.flush()
        print("\nDownload completed!")
    except Exception as e:
        print(f"\nError downloading file: {str(e)}")
        return False
    return True

def extract_bz2(filename):
    """Extract a bz2 file"""
    print(f"Extracting {filename}...")
    try:
        with bz2.BZ2File(filename, 'rb') as source, open(filename[:-4], 'wb') as dest:
            shutil.copyfileobj(source, dest)
        print("Extraction completed!")
        return True
    except Exception as e:
        print(f"Error extracting file: {str(e)}")
        return False

def setup_models(force_download=False):
    """Setup required model files"""
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Download and extract shape predictor
    predictor_bz2 = os.path.join(models_dir, 'shape_predictor_68_face_landmarks.dat.bz2')
    predictor_dat = os.path.join(models_dir, 'shape_predictor_68_face_landmarks.dat')
    
    if not os.path.exists(predictor_dat) or force_download:
        if os.path.exists(predictor_dat) and not force_download:
            print("Shape predictor model file already exists.")
            return True
            
        print("Setting up dlib shape predictor model...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        
        if download_file(url, predictor_bz2):
            if extract_bz2(predictor_bz2):
                # Clean up the bz2 file
                try:
                    os.remove(predictor_bz2)
                    print("Setup completed successfully!")
                except:
                    print("Warning: Could not remove temporary file, but setup was successful.")
            else:
                print("Error: Failed to extract the model file.")
                return False
        else:
            print("Error: Failed to download the model file.")
            return False
    else:
        # Only print this message once
        if not hasattr(setup_models, '_model_exists_printed'):
            print("Shape predictor model file already exists.")
            setup_models._model_exists_printed = True
    
    return True

if __name__ == "__main__":
    if setup_models():
        print("\nAll models are set up and ready to use!")
    else:
        print("\nError setting up models. Please check the error messages above.") 