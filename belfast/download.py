import os
import requests
import urllib3
import gzip
import shutil
import platform
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# Suppress SSL certificate warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def initialize_selenium():
    # Initialize selenium (You only need to run this cell once.)

    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no UI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    # Launch Chrome browser in headless mode
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    print("Browser launched successfully!")

    # Close the browser
    driver.quit()


def find_available_filters(web_link="https://stev.oapd.inaf.it/cgi-bin/cmd", contain_str=None):
    """
    To find all available filters of PARSEC isochrones:
    Open the web page with Selenium, find the <select name="photsys_file"> element,
    and print all available <option> values and texts.

    Parameters
    ----------
    web_link : str
        The URL of the target webpage (default: CMD 3.8 tool).
    contain_str : str or None
        Custom string for filtering output. If provided, only options whose text
        contains this string will be printed (case-insensitive).
    """

    # Configure Chrome in headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")           # no UI
    chrome_options.add_argument("--no-sandbox")         # sandbox off (Linux servers often need this)
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    # Initialize Chrome WebDriver with headless options
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Open the target web page
        driver.get(web_link)

        # Locate the <select> element by its name
        photsys_file = driver.find_element(By.NAME, "photsys_file")

        # Get all <option> elements inside <select>
        photsys_file_options = photsys_file.find_elements(By.TAG_NAME, "option")

        # Extract value and text for each option
        result = [(opt.get_attribute("value"), opt.text) for opt in photsys_file_options]

        # Print results, applying filter if provided (case-insensitive)
        for value, text in result:
            if contain_str is None or contain_str.lower() in text.lower():
                print(f"Filter: {text}, \n\tfilter_variable = {value}")
    finally:
        driver.quit()




def download_isochrone(
    download_dir,
    omega_ini,
    age_type,
    age_initial,
    age_end,
    age_interval,
    metallicity_type,
    metallicity_initial,
    metallicity_end,
    metallicity_interval,
    name_suffix=None,
    web_link = "https://stev.oapd.inaf.it/cgi-bin/cmd",
    filter_variable='YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat',
    wait_time=120,
    output_name=None,
    overwrite=False,
    unzip_flag=False,
    keep_gz=True,
):
    """
        Download stellar isochrone tables from the CMD 3.7 web interface (PARSEC v2.0) using Selenium.

        This function automates filling the CMD web form (rotation, age range, metallicity range,
        and photometric system), submits the request, and retrieves the generated isochrone file.
        The downloaded file is saved in a structured folder under `download_dir`, optionally
        decompressed if it is in `.dat.gz` format.

        Parameters
        ----------
        download_dir : str
            Directory where results will be stored. A subfolder will be created per configuration.
        omega_ini : float
            Stellar rotation parameter (0.0–0.95).
        age_type : str
            Age scale, either "linear" or "log".
        age_initial : str or float
            Lower bound of the age range (e.g., "1e6").
        age_end : str or float
            Upper bound of the age range (e.g., "1e9").
        age_interval : str or float
            Step size in age (linear or logarithmic depending on `age_type`).
        metallicity_type : str
            Metallicity definition, either "Z" (absolute) or "MH" (logarithmic [M/H]).
        metallicity_initial : float
            Lower bound of metallicity range.
        metallicity_end : float
            Upper bound of metallicity range.
        metallicity_interval : float
            Step size for metallicity.
        name_suffix : str, optional
            Custom label appended to folder and filename.
        web_link : str, optional
            CMD web interface URL (default: https://stev.oapd.inaf.it/cgi-bin/cmd).
        filter_variable : str, optional
            Photometric system identifier (must match CMD option values).
        wait_time : int, optional
            Maximum time (seconds) to wait for page elements and downloads (default: 60).
        output_name : str, optional
            Custom output filename (without extension). If None, generated automatically.
        overwrite : bool, optional
            If False (default), skip download if file already exists.
        keep_gz : bool, optional
            If False, delete the `.dat.gz` after extraction (default: True).

        Notes
        -----
        - Chrome + Selenium is required (webdriver installed automatically).
        - The function disables SSL verification for file downloads.
        - Folder and filenames are automatically constructed from age/metallicity ranges,
          unless overridden by `output_name`.
        """

    def overwrite_input(element, value):
        """Clear and reliably overwrite input content.

        On macOS use COMMAND+A, on Linux/Windows use CONTROL+A.
        """
        element.click()
        if platform.system() == "Darwin":  # macOS
            element.send_keys(Keys.COMMAND, "a")
        else:  # Linux / Windows
            element.send_keys(Keys.CONTROL, "a")
        element.send_keys(str(value))

    if name_suffix is not None:
        folder_name = f'{age_initial}_{age_end}yr_{age_type}_isochrones ({name_suffix})'
    else:
        folder_name = f'{age_initial}_{age_end}yr_{age_type}_isochrones'

    folder_path = os.path.join(download_dir, folder_name)

    if output_name is None:
        if name_suffix is not None:
            output_name = (
                f"omega_ini{omega_ini:.2f}_{age_type}_age{age_initial}to{age_end}d{age_interval}_"
                f"{metallicity_type}{metallicity_initial}to{metallicity_end}d{metallicity_interval} ({name_suffix})"
            )
        else:
            output_name = (
                f"omega_ini{omega_ini:.2f}_{age_type}_age{age_initial}to{age_end}d{age_interval}_"
                f"{metallicity_type}{metallicity_initial}to{metallicity_end}d{metallicity_interval}"
            )

    os.makedirs(folder_path, exist_ok=True)

    # Existing files
    existing_files = os.listdir(folder_path)

    # Strip .dat and .dat.gz extensions
    existing_basenames = {
        fname.replace(".dat", "").replace(".gz", "")
        for fname in existing_files
        if fname.endswith(".dat") or fname.endswith(".dat.gz")
    }

    # Check if output_name already exists (regardless of extension)
    if output_name in existing_basenames and not overwrite:
        print(f"⚠️ File already exists in directory: {output_name}.dat or .dat.gz")
        print("⏩ Skipping submission (overwrite=False)")
        return

    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no UI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    prefs = {
        "download.default_directory": folder_path,
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    chrome_options.add_experimental_option("prefs", prefs)

    # Launch Chrome browser
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    wait = WebDriverWait(driver, wait_time)  # extended timeout

    try:
        driver.get(web_link)

        # Select PARSEC V2.0 model
        wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//input[@name='track_parsec' and @value='parsec_CAF09_v2.0']")
            )
        ).click()

        # Input omega_ini
        omega_ini_input = wait.until(EC.presence_of_element_located((By.NAME, "track_omegai")))
        overwrite_input(omega_ini_input, omega_ini)

        # Select photometric system
        select = Select(wait.until(EC.presence_of_element_located((By.NAME, "photsys_file"))))
        select.select_by_value(filter_variable)

        # Input age range
        if age_type == "linear":
            wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//input[@name='isoc_isagelog' and @value='0']")
                )
            ).click()

            overwrite_input(
                wait.until(EC.presence_of_element_located((By.NAME, "isoc_agelow"))), age_initial
            )
            overwrite_input(
                wait.until(EC.presence_of_element_located((By.NAME, "isoc_ageupp"))), age_end
            )
            overwrite_input(
                wait.until(EC.presence_of_element_located((By.NAME, "isoc_dage"))), age_interval
            )

        elif age_type == "log":
            wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//input[@name='isoc_isagelog' and @value='1']")
                )
            ).click()

            overwrite_input(
                wait.until(EC.presence_of_element_located((By.NAME, "isoc_lagelow"))), age_initial
            )
            overwrite_input(
                wait.until(EC.presence_of_element_located((By.NAME, "isoc_lageupp"))), age_end
            )
            overwrite_input(
                wait.until(EC.presence_of_element_located((By.NAME, "isoc_dlage"))), age_interval
            )

        else:
            raise ValueError("age_type must be either 'linear' or 'log'")

        # Input metallicity
        if metallicity_type == "Z":
            wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//input[@name='isoc_ismetlog' and @value='0']")
                )
            ).click()

            overwrite_input(
                wait.until(EC.presence_of_element_located((By.NAME, "isoc_zlow"))),
                metallicity_initial,
            )
            overwrite_input(
                wait.until(EC.presence_of_element_located((By.NAME, "isoc_zupp"))),
                metallicity_end,
            )
            overwrite_input(
                wait.until(EC.presence_of_element_located((By.NAME, "isoc_dz"))),
                metallicity_interval,
            )

        elif metallicity_type == "MH":
            wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//input[@name='isoc_ismetlog' and @value='1']")
                )
            ).click()

            overwrite_input(
                wait.until(EC.presence_of_element_located((By.NAME, "isoc_metlow"))),
                metallicity_initial,
            )
            overwrite_input(
                wait.until(EC.presence_of_element_located((By.NAME, "isoc_metupp"))),
                metallicity_end,
            )
            overwrite_input(
                wait.until(EC.presence_of_element_located((By.NAME, "isoc_dmet"))),
                metallicity_interval,
            )

        else:
            raise ValueError("metallicity_type must be either 'Z' or 'MH'")

        # Submit form
        wait.until(EC.element_to_be_clickable((By.NAME, "submit_form"))).click()

        # Wait for .dat/.dat.gz download link
        link_element = wait.until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '.dat')]"))
        )

        relative_href = link_element.get_attribute("href")
        basename = ".dat.gz" if "gz" in relative_href else ".dat"
        full_url = relative_href.replace("../", "")

        # Download the file using requests (SSL verification disabled)
        response = requests.get(full_url, verify=False)
        save_path = os.path.join(folder_path, output_name + basename)
        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"✅ File saved to: {os.path.abspath(save_path)}")

        # 🔽 unzip (if ending with .dat.gz)
        if save_path.endswith(".dat.gz") and unzip_flag:
            extracted_path = save_path.replace(".gz", "")
            with gzip.open(save_path, "rb") as f_in:
                with open(extracted_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"📂 Extracted to: {extracted_path}")

            # delete .gz files?
            if not keep_gz:
                os.remove(save_path)
                print(f"🗑️ Removed compressed file: {save_path}")

    finally:
        driver.quit()