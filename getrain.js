// scrape_rainfall.js

// Import the Playwright library
const { chromium } = require('playwright');
const path = require('path'); // Node.js built-in module for path manipulation
const fs = require('fs');     // Node.js built-in module for file system operations

/**
 * Scrapes rainfall data from IMD Pune website for a specified year and saves it.
 * @param {number} yearToSelect - The year to select from the dropdown (e.g., 2023).
 * @param {string} outputFileName - The desired name for the downloaded file (e.g., 'rainfall_2023.nc').
 * @param {string} outputDirectory - The directory where the file should be saved (e.g., 'rain_data').
 */
async function scrapeRainfallData(yearToSelect, outputFileName, outputDirectory) {
    let browser; // Declare browser variable outside try-catch for finally block access

    // Define the full path where the file should be saved, including the directory
    const saveDirectory = path.join(__dirname, outputDirectory);
    const savePath = path.join(saveDirectory, outputFileName);

    // Check if the file already exists
    if (fs.existsSync(savePath)) {
        console.log(`Skipping year ${yearToSelect}: File already exists at ${savePath}`);
        return; // Exit the function if the file exists
    }

    try {
        // Ensure the output directory exists. If not, create it.
        if (!fs.existsSync(saveDirectory)) {
            console.log(`Creating directory: ${saveDirectory}`);
            fs.mkdirSync(saveDirectory, { recursive: true });
        }

        // Launch a Chromium browser instance in headless mode (true by default).
        // Set headless to false if you want to see the browser UI during execution.
        browser = await chromium.launch({ headless: true });
        
        // Create a new browser context. This isolates browser sessions.
        const context = await browser.newContext();
        
        // Create a new page within the context
        const page = await context.newPage();

        // Navigate to the target URL
        console.log(`Navigating to https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html for year ${yearToSelect}`);
        await page.goto('https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html', {
            waitUntil: 'domcontentloaded' // Wait until the DOM is loaded
        });
        console.log('Page loaded successfully.');

        // The dropdown has an ID of 'RF25'.
        const yearDropdownSelector = '#RF25'; 
        
        // Wait for the dropdown element to be visible on the page
        console.log(`Waiting for dropdown with selector '${yearDropdownSelector}' to be visible...`);
        await page.waitForSelector(yearDropdownSelector, { state: 'visible', timeout: 30000 }); // Wait up to 30 seconds
        console.log('Dropdown is visible.');

        // Select the option by its value (which is the year itself)
        console.log(`Attempting to select year: ${yearToSelect}`);
        await page.selectOption(yearDropdownSelector, String(yearToSelect));
        console.log(`Selected year: ${yearToSelect}`);

        // The download button has a value attribute of 'Download'.
        const downloadButtonSelector = 'input[type="submit"][value="Download"]'; 
        
        // Wait for the download button to be visible on the page
        console.log(`Waiting for download button with selector '${downloadButtonSelector}' to be visible...`);
        await page.waitForSelector(downloadButtonSelector, { state: 'visible', timeout: 30000 });
        console.log('Download button is visible.');

        // Playwright's way to handle downloads:
        // Start waiting for the download before clicking the button.
        // This ensures that the download event is captured.
        console.log('Waiting for download to start...');
        const [download] = await Promise.all([
            page.waitForEvent('download'), // Wait for the download event
            page.click(downloadButtonSelector) // Click the download button
        ]);

        // Get the suggested filename from the download event (optional, for logging)
        const suggestedFileName = download.suggestedFilename();
        console.log(`Download initiated. Suggested filename: ${suggestedFileName}`);
        
        // Save the downloaded file to the specified path
        await download.saveAs(savePath);
        console.log(`File saved successfully to: ${savePath}`);

    } catch (error) {
        console.error(`An error occurred during scraping for year ${yearToSelect}:`, error);
    } finally {
        // Ensure the browser is closed even if an error occurs
        if (browser) {
            await browser.close();
            console.log('Browser closed.');
        }
    }
}

// --- Configuration for the loop ---
const START_YEAR = 1901;
const END_YEAR = 2024;
const OUTPUT_FOLDER = 'rain_data'; // The folder where files will be saved
const DELAY_BETWEEN_DOWNLOADS_MS = 2000; // 2 seconds delay to be polite to the server

/**
 * Helper function to introduce a delay.
 * @param {number} ms - The delay in milliseconds.
 */
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Main execution loop
(async () => {
    console.log(`Starting data download from ${START_YEAR} to ${END_YEAR}...`);
    for (let year = START_YEAR; year <= END_YEAR; year++) {
        const fileName = `rain_0p25_${year}.nc`; // Custom filename based on your request
        await scrapeRainfallData(year, fileName, OUTPUT_FOLDER);
        
        // Add a delay before the next iteration, unless it's the last year
        if (year < END_YEAR) {
            console.log(`Waiting for ${DELAY_BETWEEN_DOWNLOADS_MS / 1000} seconds before next download...`);
            await delay(DELAY_BETWEEN_DOWNLOADS_MS);
        }
    }
    console.log('All requested years processed.');
})();