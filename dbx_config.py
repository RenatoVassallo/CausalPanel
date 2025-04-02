import pandas as pd
import io
import time
import pickle
import os
import gc
from tqdm import tqdm
from dotenv import load_dotenv
import requests
#from inst_disr.pipeline.config.metadata import today
import dropbox

class DropboxHelper:
    """
    This class contains helper functions for reading and writing files to Dropbox.

    Args:
    - dbx_token (str): The Dropbox API token.
    - dbx_key (str): The Dropbox app key.
    - dbx_secret (str): The Dropbox app secret.
    - input_path (str): The base path in Dropbox where the input data is stored.
    - output_path (str): The base path in Dropbox where the output data will be stored.
    - custom_paths (bool): If True, the input_path and output_path will be used as is. If False, the input_path and output_path will be used to create the raw_input_path, clean_input_path and output_path.

    Attributes:
    - dbx (dropbox.Dropbox): The Dropbox object used to interact with the Dropbox API.
    - raw_input_path (str): The path of the Dropbox folder containing the raw data.
    - clean_input_path (str): The path of the Dropbox folder containing the clean data.
    - output_path (str): The path of the Dropbox folder where the output data will be saved.
    
    """
    
    def __init__(self, dbx_token:str, dbx_key:str, dbx_secret:str, input_path:str, output_path:str, custom_paths:bool = False):

        self.dbx = dropbox.Dropbox(oauth2_refresh_token = dbx_token, app_key = dbx_key, app_secret = dbx_secret)

        if custom_paths:
            self.input_path = input_path
            self.output_path = output_path
        
        else:
            self.raw_input_path, self.clean_input_path, self.output_path = self._initialize_paths(input_path, output_path)

    def _initialize_paths(self, input_path:str, output_path:str):

        """
        Function to initialize your Dropbox App and create the necessary folders.

        Args
        ----

        input_path (str): The base path in Dropbox where the input data is stored.
        output_path (str): The base path in Dropbox where the output data will be stored.

        Returns
        -------
        raw_input_path (str): The path of the Dropbox folder containing the raw data.
        clean_input_path (str): The path of the Dropbox folder containing the clean data.
        output_path (str): The path of the Dropbox folder where outputs will be saved.
        """

        raw_input_path = f"{input_path}/raw"
        clean_input_path = f"{input_path}/clean"
        output_path = output_path

        #If the folder doesn't exist (i.e. you didn't manually create your input/raw, input/clean and output folders in Dropbox), then make them
        for path in [raw_input_path, clean_input_path, output_path]:
            if not self.folder_exists(path):
                self.create_folder(path)
            else:
                pass

        return raw_input_path, clean_input_path, output_path

    """
    Directory and file management functions
    """

    def folder_exists(self, folder_path):

        # Check if the folder exists
        try:
            self.dbx.files_get_metadata(folder_path)
            return True
        except dropbox.exceptions.ApiError as err:
            if isinstance(err.error, dropbox.files.GetMetadataError) and err.error.is_path() and \
                    err.error.get_path().is_not_found():
                return False
            else:
                print(f"Failed to check if folder '{folder_path}' exists:", err)
                
                
    def file_exists(self, file_path):
        """
        Check if a file exists in Dropbox.

        Args:
        - file_path (str): The full Dropbox path to the file.

        Returns:
        - bool: True if the file exists, False otherwise.
        """
        try:
            # Attempt to retrieve metadata for the file
            metadata = self.dbx.files_get_metadata(file_path)
            if isinstance(metadata, dropbox.files.FileMetadata):
                print(f"File '{file_path}' exists.")
                return True
            return False
        except dropbox.exceptions.ApiError as err:
            if isinstance(err.error, dropbox.files.GetMetadataError) and err.error.is_path() and \
                    err.error.get_path().is_not_found():
                print(f"File '{file_path}' does not exist.")
                return False
            else:
                print(f"Failed to check if file '{file_path}' exists:", err)
                raise


    def create_folder(self, folder_path, return_path = False):
    
        # Try to create the new folder
        try:
            self.dbx.files_create_folder_v2(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except dropbox.exceptions.ApiError as err:
            # Check if the error is because the folder already exists
            if isinstance(err.error, dropbox.files.CreateFolderError) and err.error.is_path() and \
                    err.error.get_path().is_conflict():
                print(f"Folder '{folder_path}' already exists.")
            else:
                print(f"Failed to create folder '{folder_path}':", err)

        if return_path:
            return folder_path
        
    def list_files_in_folder(self, folder_path):
    # List all files in the folder
        try:
            files = []
            result = self.dbx.files_list_folder(folder_path)

            # Append initial entries
            files.extend(result.entries)

            # Continue fetching if there are more entries
            while result.has_more:
                result = self.dbx.files_list_folder_continue(result.cursor)
                files.extend(result.entries)

            print(f"Files in folder '{folder_path}':")
            return [f.name for f in files]

        except dropbox.exceptions.ApiError as err:
            print(f"Failed to list files in folder '{folder_path}':", err)
        
    """
    Reading functions
    """
    def get_file_size(self, dbx_path: str, directory: str, filename: str) -> int:
        """
        Retrieves the size of a file in Dropbox.

        Args:
        - dbx_path (str): The base Dropbox path where the file is stored.
        - directory (str): The directory within the base path where the file is stored.
        - filename (str): The name of the file (e.g., 'my_file.csv').

        Returns:
        - int: The size of the file in bytes.
        """
        # Full path to the file in Dropbox
        file_path = f'{dbx_path}/{directory}/{filename}'
        
        try:
            # Retrieve file metadata
            metadata = self.dbx.files_get_metadata(file_path)
            # Extract file size from metadata
            file_size = metadata.size
            print(f"Size of '{filename}': {file_size} bytes")
            return file_size
        except dropbox.exceptions.ApiError as e:
            print(f"Failed to retrieve size for '{filename}'. Error: {e}")
            return -1  # Return -1 or an appropriate value if the file size retrieval fails
        
                    
    def read_csv(self, dbx_path: str, directory: str, filename: str, skiprows=None, 
                 usecols=None, sep=',', index_col=None, parse_dates=None, 
                 dtype=None, chunk_size=None, year_month_combos=None, skip_chunk=None):
        """
        Reads a CSV file from Dropbox into a pandas DataFrame.

        Args:
        - dbx_path (str): The base Dropbox path.
        - directory (str): Directory containing the file.
        - filename (str): The file name.
        - skiprows (int, optional): Number of rows to skip at the beginning of the file.
        - usecols (list, optional): Columns to read; reads all if None.
        - sep (str, optional): Delimiter of the CSV. Default is ','.
        - index_col (int or str, optional): Column to use as the row labels of the DataFrame.
        - parse_dates (list or bool, optional): Columns to parse as dates.
        - dtype (dict, optional): Data types of the columns.
        - chunk_size (int, optional): File chunk size in bytes for partial reading.
        - year_month_combos (list, optional): List of (month, year) tuples to filter data.
        - skip_chunk (list or range, optional): List or range of chunk indices to skip. Default is None.

        Returns:
        - pd.DataFrame: The resulting DataFrame.
        """
        file_path = f'{dbx_path}/{directory}/{filename}'

        # Download the entire file if chunk_size is not specified
        if chunk_size is None:
            _, res = self.dbx.files_download(file_path)
            df = pd.read_csv(
                io.StringIO(res.content.decode('utf-8')),
                skiprows=skiprows,
                usecols=usecols,
                sep=sep,
                index_col=index_col,
                parse_dates=parse_dates,
                dtype=dtype
            )
            return df
        
        # Prepare chunk-based processing
        else:
            
            if year_month_combos is None:
                today = pd.Timestamp.today()
                year_month_combos = [
                    (date.month, date.year) for date in pd.date_range(
                        start='1989-01-01', end=today, freq='MS'
                    )
                ]
                print("Using default year_month_combos from '1989-01-01' to today.")

            periods_to_keep = [f"{year}{month:02}" for month, year in year_month_combos]
            size = self.get_file_size(dbx_path, directory, filename)
            chunks = [(i * chunk_size, min(size, (i + 1) * chunk_size - 1)) for i in range((size + chunk_size - 1) // chunk_size)]

            metadata = self.dbx.files_get_temporary_link(file_path)
            temp_link = metadata.link
            last_line = ""
            dfs = []

            # Default to no skipping if skip_chunk is None
            if skip_chunk is None:
                skip_chunk = []

            # Step 4: Process each chunk
            for chunk_index, (start, end) in enumerate(tqdm(chunks, desc="Processing chunks")):
                try:
                    # Skip chunks as per skip_chunk parameter
                    if chunk_index in skip_chunk:
                        print(f"Skipping chunk {chunk_index} ({start}-{end})")
                        continue
                
                    headers = {"Range": f"bytes={start}-{end}"}
                    res = requests.get(temp_link, headers=headers)
                    
                    if res.status_code != 206:
                        print(f"Warning: Expected partial content (206), got {res.status_code}.")
                        continue

                    content_str = res.content.decode('utf-8')

                    # Extract column names
                    if start == 0:
                        col_names = content_str.splitlines()[0].split(',')
                        content_str = "\n".join(content_str.splitlines()[1:])
                    else:
                        content_str = last_line + content_str

                    content_lines = content_str.splitlines()

                    # Process content line by line for valid DataFrame parsing
                    for i in reversed(range(len(content_lines))):
                        try:
                            content_try = content_lines[:i]
                            df_try = pd.read_csv(
                                io.StringIO("\n".join(content_try)),
                                skiprows=skiprows,
                                usecols=usecols,
                                sep=sep,
                                index_col=index_col,
                                parse_dates=parse_dates,
                                dtype=dtype,
                                names=col_names
                            )

                            # Handle 'yyyymm' creation robustly
                            df_try = self._create_yyyymm_column(df_try)

                            # Filter DataFrame by periods
                            df_filtered = df_try[df_try['yyyymm'].isin(periods_to_keep)].drop(columns='yyyymm')
                            if not df_filtered.empty:
                                dfs.append(df_filtered)

                            last_line = "".join(content_lines[i:])
                            break
                        except pd.errors.ParserError:
                            continue
                except Exception as e:
                    print(f"Error while processing chunk {start}-{end}: {e}")
                finally:
                    gc.collect()

            # Combine all DataFrames
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            else:
                print("No data was successfully parsed.")
                return None

    def _create_yyyymm_column(self, df):
        """
        Creates a 'yyyymm' column from a 'date' or 'period' column in a DataFrame.
        Args:
        - df (pd.DataFrame): The DataFrame.
        Returns:
        - pd.DataFrame: The DataFrame with a 'yyyymm' column.
        """
        # Determine the column to use, prioritizing 'period' over 'date'
        if 'period' in df.columns:
            date_col = 'period'
        elif 'date' in df.columns:
            date_col = 'date'
        else:
            raise KeyError("Neither 'period' nor 'date' column is present in the DataFrame.")
        
        try:
            # Attempt to create 'yyyymm' using string slicing
            df['yyyymm'] = df[date_col].astype(str).str[:6]
            if df['yyyymm'].isnull().all() or df['yyyymm'].empty:
                raise ValueError(f"Column 'yyyymm' is empty after processing '{date_col}' as string.")
        except (KeyError, ValueError, AttributeError):
            # Fall back to parsing dates and formatting
            df['yyyymm'] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y%m')
            if df['yyyymm'].isnull().all():
                raise ValueError(f"Column 'yyyymm' is empty after attempting to parse '{date_col}' as datetime.")
        
        return df

        
        
    def read_pickle(self, dbx_path:str, directory:str, filename:str):
        """
        Downloads a pickle file from Dropbox, deserializes it, and returns the Python object.

        Args:
        - dbx_path (str): The base Dropbox path where the file is stored.
        - directory (str): The directory within the base path where the file is stored.
        - filename (str): The name of the pickle file (e.g., 'my_object.pkl').

        Returns:
        The deserialized Python object from the pickle file.
        """
        # Full path where the file is stored in Dropbox
        file_path = f'{dbx_path}/{directory}/{filename}'

        try:
            # Use Dropbox's files_download to get the file
            _, res = self.dbx.files_download(file_path)
            
            # Deserialize the pickle data from the response content
            obj = pickle.loads(res.content)
            
            return obj
        except Exception as e:
            print(f"Failed to load '{filename}' from Dropbox. Error: {e}")
            return None
        
    def read_excel(self, dbx_path:str, directory:str, filename:str, sheet_name:str = None, header:int = 0):
        """
        This function reads an Excel file from Dropbox into a pandas DataFrame.

        Args:
        - file_path (str): The path of the Excel file in Dropbox.
        - sheet_name (str): The name of the sheet to read from the Excel file. If None, the first sheet will be read by default.

        Returns:
        - df (pandas.DataFrame): The DataFrame containing the data from the Excel file.
        """

        full_dropbox_path = f'{dbx_path}/{directory}/{filename}'
        try:
            # Use Dropbox's files_download to get the file
            _, res = self.dbx.files_download(full_dropbox_path)
            
            # Read the specified sheet into a pandas DataFrame
            # If sheet_name is None, pd.read_excel will read the first sheet by default
            df = pd.read_excel(io.BytesIO(res.content), sheet_name=sheet_name,header=header)
            
            return df
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    """
    Writing functions
    """

    def write_csv(self, df: pd.DataFrame, dbx_path:str, directory:str, filename:str, print_success = True, print_size = True):
        """
        Saves a DataFrame to a CSV file and uploads it to Dropbox.

        Args:
        - df (pandas.DataFrame): The DataFrame to save.
        - write_path (str): The path in Dropbox where the file will be saved.
        - filename (str): The name of the file (e.g., 'my_dataframe.csv').
        """

        # Convert DataFrame to CSV format
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue().encode('utf-8')

        size_in_bytes = len(csv_content)
        size_in_mb = size_in_bytes / 1024**2   # Convert kilobytes to megabytes

        if print_size:
            print(f"Size of the CSV file: {size_in_mb:.2f} MB")

        
        # Full path where the file will be saved in Dropbox
        full_dropbox_path = f'{dbx_path}/{directory}/{filename}'

        # Upload the CSV to Dropbox
        try:
            if size_in_mb < 150: #dropbox API has a limit of 150MB for this method
                self.dbx.files_upload(csv_content, full_dropbox_path, mode=dropbox.files.WriteMode.overwrite)

                if print_success:
                    print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")

            else:
                # Use chunked upload for large files
                CHUNK_SIZE = 25 * 1024 * 1024  # 25MB chunk size
                upload_session_start_result = self.dbx.files_upload_session_start(csv_content[:CHUNK_SIZE])
                cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                                        offset=csv_content[:CHUNK_SIZE].__len__())
                remaining_content = csv_content[CHUNK_SIZE:]
                
                while len(remaining_content) > 0:
                    if len(remaining_content) > CHUNK_SIZE:
                        self.dbx.files_upload_session_append_v2(remaining_content[:CHUNK_SIZE], cursor)
                        cursor.offset += CHUNK_SIZE
                        remaining_content = remaining_content[CHUNK_SIZE:]
                    else:
                        # Move the remaining data in the last chunk
                        self.dbx.files_upload_session_finish(remaining_content, cursor,
                                                            dropbox.files.CommitInfo(path=full_dropbox_path,
                                                                                    mode=dropbox.files.WriteMode.overwrite))
                        remaining_content = []
                if print_success:
                    print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
            
        except Exception as e:
            print(f"Failed to upload '{filename}' to Dropbox. Error: {e}")

    def write_pickle(self, obj, dbx_path:str, directory:str, filename:str):
        """
        Serializes a Python object using pickle and uploads it to Dropbox.

        Args:
        - obj (Any): The Python object to serialize and save.
        - dbx_path (str): The base Dropbox path where the file will be saved.
        - directory (str): The directory within the base path where the file will be saved.
        - filename (str): The name of the file (e.g., 'my_object.pkl').
        """
        # Serialize the object to a bytes stream
        pickle_buffer = io.BytesIO()
        pickle.dump(obj, pickle_buffer)
        pickle_buffer.seek(0)  # Rewind the buffer to the beginning after writing

        # Full path where the file will be saved in Dropbox
        full_dropbox_path = f'{dbx_path}/{directory}/{filename}'

        # Upload the serialized object to Dropbox
        try:
            self.dbx.files_upload(pickle_buffer.getvalue(), full_dropbox_path, mode=dropbox.files.WriteMode.overwrite)
            print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
        except Exception as e:
            print(f"Failed to upload '{filename}' to Dropbox. Error: {e}")

    def write_fig(self, buffer, dbx_path:str, directory:str, filename:str):
        """
        Uploads a PNG image from an in-memory buffer to Dropbox.

        Args:
        - buffer (io.BytesIO): The buffer containing the PNG image data.
        - directory (str): The name of the directory where the file will be saved.
        - filename (str): The name of the file (e.g., 'my_plot.png').
        - write_path (str): The path in Dropbox where the file will be saved.
        """
        
        # Full path where the file will be saved in Dropbox
        full_dropbox_path = f'{dbx_path}/{directory}/{filename}'

        # Upload the PNG to Dropbox
        try:
            self.dbx.files_upload(buffer.getvalue(), full_dropbox_path, mode=dropbox.files.WriteMode.overwrite)
            print(f"File '{filename}' successfully uploaded to Dropbox path: '{full_dropbox_path}'")
        except Exception as e:
            print(f"Failed to upload '{filename}' to Dropbox. Error: {e}")

load_dotenv(override = True)
dbx_helper = DropboxHelper(
    dbx_token = os.getenv('DROPBOX_TOKEN'),
    dbx_key = os.getenv('DROPBOX_KEY'),
    dbx_secret= os.getenv('DROPBOX_SECRET'),
    input_path = os.getenv('INPUT_PATH'),
    output_path = os.getenv('OUTPUT_PATH'),
    custom_paths = False
    )

cf_dbx_helper = DropboxHelper(
        dbx_token = os.getenv('CF_DROPBOX_TOKEN'),
        dbx_key = os.getenv('CF_DROPBOX_KEY'),
        dbx_secret= os.getenv('CF_DROPBOX_SECRET'),
        input_path = f"{os.getenv('CF_INPUT_PATH')}",
        output_path = os.getenv('CF_OUTPUT_PATH'),
        custom_paths=True
    )