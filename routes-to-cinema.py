import pandas as pd
import requests
import os
import time
import logging
import json
import re
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (for API key)
load_dotenv()

def format_address(address):
    """
    Format address to ensure it works with the Google API.
    For UK postcodes, ensure they are properly formatted.
    
    Args:
        address (str): Original address or postcode
        
    Returns:
        str: Formatted address
    """
    # Strip whitespace
    address = address.strip()
    
    # Check if it's likely a UK postcode (basic check)
    uk_postcode = False
    if len(address) <= 10 and " " in address:
        uk_postcode = True
    
    # If it looks like a UK postcode and doesn't already have UK in it
    if uk_postcode and "UK" not in address and "United Kingdom" not in address:
        return f"{address}, UK"
    
    return address

def get_route_details(origin, destination, mode, specific=None):
    """
    Get route details using Google Routes API
    
    Args:
        origin (str): Origin location (postcode)
        destination (str): Destination location (postcode)
        mode (str): Transportation mode (driving, walking, bicycling, transit)
        specific (str, optional): Specific transit mode when mode is transit
    
    Returns:
        dict: Dictionary containing step methods and distances
    """
    # Format addresses
    origin = format_address(origin)
    destination = format_address(destination)
    
    # Get API key from environment variable
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logger.error("Google Maps API key not found. Please set GOOGLE_MAPS_API_KEY in .env file")
        return {"Method1": "Not found - API key missing", "Distance1": "Not found"}
    
    # Log the request details
    logger.info(f"Getting route details from '{origin}' to '{destination}' via '{mode}'")
    if specific and mode.lower() == "transit":
        logger.info(f"Specific transit mode: {specific}")
    
    # Validate mode and convert to Google's format
    valid_modes = {
        "driving": "DRIVE",
        "walking": "WALK", 
        "bicycling": "BICYCLE",
        "bicycle": "BICYCLE",
        "transit": "TRANSIT",
        "car": "DRIVE",
        "bus": "TRANSIT",
        "train": "TRANSIT",
        "tube": "TRANSIT"
    }
    
    google_mode = valid_modes.get(mode.lower(), "DRIVE")
    logger.info(f"Using travel mode: {google_mode}")
    
    # Define the API endpoint
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    
    # Define headers with correct field mask for Routes API V2
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "routes,routes.legs,routes.legs.steps,routes.legs.steps.distanceMeters,routes.legs.steps.travelMode,routes.legs.steps.transitDetails,routes.legs.steps.transitDetails.stopDetails,routes.legs.steps.transitDetails.localizedValues,routes.legs.steps.transitDetails.transitLine,routes.legs.steps.transitDetails.transitLine.agencies,routes.legs.steps.transitDetails.transitLine.name,routes.legs.steps.transitDetails.transitLine.vehicle"
    }
    
    # Define request body according to Routes API V2 requirements
    data = {
        "origin": {
            "address": origin
        },
        "destination": {
            "address": destination
        },
        "travelMode": google_mode,
        "routingPreference": "ROUTING_PREFERENCE_UNSPECIFIED",
        "computeAlternativeRoutes": False,
        "languageCode": "en-GB"  # Using en-GB for UK addresses
    }
    
    # Add transitPreferences if mode is transit and specific mode is provided
    if google_mode == "TRANSIT" and specific:
        # Map specific transit modes to Google's format
        transit_modes_map = {
            "bus": "BUS",
            "subway": "SUBWAY",
            "train": "TRAIN",
            "rail": "TRAIN",
            "light_rail": "LIGHT_RAIL",
            "tram": "TRAM",
            "metro": "SUBWAY",
            "underground": "SUBWAY",
            "tube": "SUBWAY"
        }
        
        transit_mode = transit_modes_map.get(specific.lower(), "TRANSIT_TRAVEL_MODE_UNSPECIFIED")
        logger.info(f"Using specific transit mode: {transit_mode}")
        
        data["transitPreferences"] = {
            "allowedTravelModes": [transit_mode]
        }

    
    try:
        # Make API request with detailed logging
        logger.info(f"API Request URL: {url}")
        logger.info(f"API Request Headers: {headers}")
        logger.info(f"API Request Data: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, headers=headers, json=data)
        
        # Log response status
        logger.info(f"Response status code: {response.status_code}")
        
        # Check for HTTP errors
        if response.status_code != 200:
            error_message = f"API Error {response.status_code}"
            try:
                # Try to extract error details from the response
                error_details = response.json()
                if 'error' in error_details:
                    error_message += f": {error_details['error'].get('message', 'Unknown error')}"
                    logger.error(f"Detailed error: {error_details}")
            except:
                error_message += f": {response.text[:100]}..."
            
            logger.error(f"HTTP error: {error_message}")
            return {"Method1": error_message, "Distance1": "Not found"}
        
        # Process the response
        result = response.json()
        
        # Save the full response to a file for debugging
        with open("last_api_response.json", "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved full API response to last_api_response.json")
        
        # Check if routes exist
        if 'routes' not in result or not result['routes']:
            logger.warning("No routes found in API response")
            return {"Method1": "No routes found", "Distance1": "Not found"}
        
        # Get the first route
        route = result['routes'][0]
        
        # Get the steps from the first leg
        if 'legs' not in route or not route['legs']:
            logger.warning("No legs found in route")
            return {"Method1": "No legs found", "Distance1": "Not found"}
            
        leg = route['legs'][0]
        steps = leg.get('steps', [])
        
        # Log number of steps found
        logger.info(f"Found {len(steps)} steps in route")
        
        if not steps:
            logger.warning("No steps found in legs")
            return {"Method1": "No steps found", "Distance1": "Not found"}
        
        # Extract method and distance for each step
        step_data = {}
        for idx, step in enumerate(steps, 1):
            travel_mode = step.get('travelMode', 'UNKNOWN')
            distance_meters = step.get('distanceMeters', 0)
            
            # Convert distance to kilometers and round to 2 decimal places
            distance_km = round(distance_meters / 1000, 2)
            
            # Convert travel mode to more friendly format for output
            friendly_mode = travel_mode
            if travel_mode == "DRIVE":
                friendly_mode = "DRIVING"
            elif travel_mode == "WALK":
                friendly_mode = "WALKING"
            elif travel_mode == "BICYCLE":
                friendly_mode = "BICYCLING"
            
            # If transit, get more specific details about the vehicle type
            if travel_mode == "TRANSIT" and 'transitDetails' in step:
                transit_details = step.get('transitDetails', {})
                transit_line = "Unknown Line"
                transit_vehicle_type = "Unknown"
                
                # Try to get transit line and vehicle type information - structure depends on the actual response
                try:
                    # First try to get from transitLine
                    if 'transitLine' in transit_details:
                        transit_line = transit_details['transitLine'].get('name', 'Unknown Line')
                        
                        # Get vehicle type from transitLine.vehicle
                        if 'vehicle' in transit_details['transitLine']:
                            vehicle = transit_details['transitLine']['vehicle']
                            transit_vehicle_type = vehicle.get('type', 'Unknown')
                    
                    # If not found, try other paths
                    elif 'localizedValues' in transit_details:
                        if 'transitLine' in transit_details['localizedValues']:
                            transit_line = transit_details['localizedValues']['transitLine'].get('name', 'Unknown Line')
                    
                    # Log what we found
                    logger.info(f"Transit details - Line: {transit_line}, Vehicle: {transit_vehicle_type}")
                    
                    # Map vehicle type to friendly names
                    vehicle_types = {
                        "BUS": "BUS",
                        "SUBWAY": "SUBWAY",
                        "TRAIN": "RAIL",
                        "LIGHT_RAIL": "LIGHT RAIL",
                        "TRAM": "TRAM",
                        "METRO_RAIL": "METRO",
                        "COMMUTER_TRAIN": "TRAIN",
                        "HEAVY_RAIL": "RAIL"
                    }
                    
                    friendly_mode = vehicle_types.get(transit_vehicle_type, "TRANSIT")
                    friendly_mode = f"{friendly_mode} ({transit_line})"
                except Exception as e:
                    logger.warning(f"Error extracting transit details: {e}")
                    friendly_mode = "TRANSIT"
            
            step_data[f"Method{idx}"] = friendly_mode
            step_data[f"Distance{idx}"] = distance_km
            
            logger.info(f"Step {idx}: Method={friendly_mode}, Distance={distance_km}km")
        
        return step_data
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")
        return {"Method1": f"Request Error: {str(e)[:50]}", "Distance1": "Not found"}
    
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Error processing API response: {e}")
        return {"Method1": f"Processing Error: {str(e)[:50]}", "Distance1": "Not found"}
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"Method1": f"Unexpected Error: {str(e)[:50]}", "Distance1": "Not found"}

def test_geocoding(address):
    """
    Test if Google can geocode an address using the Geocoding API
    
    Args:
        address (str): Address to geocode
        
    Returns:
        bool: True if geocoding successful, False otherwise
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logger.error("Google Maps API key not found")
        return False
    
    # Format the address
    formatted_address = format_address(address)
    logger.info(f"Testing geocoding for address: '{formatted_address}'")
    
    # Use the Geocoding API to test if the address is valid
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={formatted_address}&key={api_key}"
    
    try:
        response = requests.get(url)
        result = response.json()
        
        if response.status_code == 200 and result['status'] == 'OK':
            location = result['results'][0]['geometry']['location']
            formatted = result['results'][0]['formatted_address']
            logger.info(f"Geocoding successful: '{formatted}' at {location}")
            return True
        else:
            logger.warning(f"Geocoding failed: {result['status']}")
            logger.warning(f"Response: {result}")
            return False
    
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return False

def process_csv(input_file='travel-to-cinema.csv', output_file='travel-to-cinema-results.csv'):
    """
    Process CSV file with origin, destination and mode columns
    
    Args:
        input_file (str): Input CSV filename
        output_file (str): Output CSV filename
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file '{input_file}' not found")
            raise FileNotFoundError(f"Input file '{input_file}' not found")
        
        # Load CSV file
        logger.info(f"Loading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Check if required columns exist
        required_columns = ['Home', 'Cinema', 'Method']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {', '.join(missing_columns)}")
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Display sample of input data
        logger.info(f"Sample data: \n{df.head()}")
        
        # Create empty list to store results
        results = []
        
        # Process each row
        for idx, row in df.iterrows():
            logger.info(f"\n--- Processing row {idx+1}/{len(df)} ---")
            
            # Get origin, destination and mode
            origin = str(row['Home']).strip()
            destination = str(row['Cinema']).strip()
            mode = str(row['Method']).strip()
            
            # Get specific if available
            specific = None
            if 'Specific' in row and not pd.isna(row['Specific']):
                specific = str(row['Specific']).strip()
            
            logger.info(f"Row {idx+1}: Origin='{origin}', Destination='{destination}', Mode='{mode}', Specific='{specific}'")
            
            # Skip if any value is missing or NaN
            if pd.isna(origin) or pd.isna(destination) or pd.isna(mode) or origin == 'nan' or destination == 'nan' or mode == 'nan':
                logger.warning(f"Missing data in row {idx+1}")
                result_row = {
                    "Method1": "Not found - Missing data",
                    "Distance1": "Not found"
                }
                results.append({**row.to_dict(), **result_row})
                continue
            
            # Test geocoding for origin and destination
            logger.info("Testing geocoding for addresses")
            origin_valid = test_geocoding(origin)
            destination_valid = test_geocoding(destination)
            
            if not origin_valid:
                logger.warning(f"Origin address '{origin}' could not be geocoded")
                result_row = {
                    "Method1": "Not found - Invalid origin address",
                    "Distance1": "Not found"
                }
                results.append({**row.to_dict(), **result_row})
                continue
            
            if not destination_valid:
                logger.warning(f"Destination address '{destination}' could not be geocoded")
                result_row = {
                    "Method1": "Not found - Invalid destination address",
                    "Distance1": "Not found"
                }
                results.append({**row.to_dict(), **result_row})
                continue
            
            # Get route details
            route_details = get_route_details(origin, destination, mode, specific)
            
            # If no steps were found, set a default "Not found"
            if not route_details:
                logger.warning(f"No route details returned for row {idx+1}")
                route_details = {
                    "Method1": "Not found - No details",
                    "Distance1": "Not found"
                }
            
            # Add route details to row data
            results.append({**row.to_dict(), **route_details})
            
            # Sleep to avoid hitting API rate limits
            logger.info("Waiting before next request...")
            time.sleep(1)
        
        # Create new dataframe
        logger.info("Creating result dataframe")
        result_df = pd.DataFrame(results)
        
        # Save to CSV
        logger.info(f"Saving results to {output_file}")
        result_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Display sample of output data
        logger.info(f"Sample output: \n{result_df.head()}")
        
        # Generate aggregate data
        logger.info("Generating aggregate data")
        generate_aggregate_data(result_df)
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise

def generate_aggregate_data(results_df, output_file='travel-to-cinema-aggregate.csv'):
    """
    Generate aggregate data summarizing distances by travel method for each journey
    
    Args:
        results_df (pandas.DataFrame): DataFrame containing route step details
        output_file (str): Output CSV filename for aggregated data
    """
    logger.info("Generating aggregate travel data")
    
    try:
        # Create a copy of the DataFrame
        agg_df = results_df.copy()
        
        # Identify all possible method columns (Method1, Method2, etc.) and distance columns (Distance1, Distance2, etc.)
        method_cols = [col for col in agg_df.columns if col.startswith('Method') and col[6:].isdigit()]
        distance_cols = [col for col in agg_df.columns if col.startswith('Distance') and col[8:].isdigit()]
        
        logger.info(f"Found {len(method_cols)} method columns and {len(distance_cols)} distance columns")
        
        # Define common transport mode types to aggregate
        transport_types = [
            'WALKING', 'DRIVING', 'BICYCLING', 'BUS', 'RAIL', 'SUBWAY', 
            'TRAM', 'LIGHT RAIL', 'METRO', 'TRANSIT'
        ]
        
        # Create columns for each transport type
        for transport_type in transport_types:
            agg_df[transport_type] = 0.0
        
        # Add a TOTAL column
        agg_df['TOTAL'] = 0.0
        
        # Process each row
        for idx, row in agg_df.iterrows():
            logger.info(f"Processing aggregate data for row {idx+1}")
            
            # Reset totals for this row
            for transport_type in transport_types:
                agg_df.at[idx, transport_type] = 0.0
            
            # Process each method/distance pair
            for i in range(1, len(method_cols) + 1):
                method_col = f"Method{i}"
                distance_col = f"Distance{i}"
                
                # Skip if either column doesn't exist
                if method_col not in agg_df.columns or distance_col not in agg_df.columns:
                    continue
                
                # Get method and distance
                method = str(row[method_col])
                distance_str = str(row[distance_col])
                
                # Skip if method or distance is not found or invalid
                if "Not found" in method or "Not found" in distance_str or method == "nan" or distance_str == "nan":
                    continue
                
                try:
                    # Extract distance as float
                    distance = float(distance_str)
                    
                    # Add to total distance
                    agg_df.at[idx, 'TOTAL'] += distance
                    
                    # Determine transport type from method string
                    transport_found = False
                    for transport_type in transport_types:
                        # Handle special case for transit with lines (e.g., "BUS (Route 123)")
                        if transport_type in method:
                            # If there's a match, add the distance to the corresponding transport type
                            agg_df.at[idx, transport_type] += distance
                            transport_found = True
                            break
                    
                    # If no specific transport type was found, add to TRANSIT as fallback
                    if not transport_found and 'TRANSIT' in method:
                        agg_df.at[idx, 'TRANSIT'] += distance
                
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing distance for row {idx+1}, method {i}: {e}")
                    continue
        
        # Create a clean aggregate DataFrame with only the columns we need
        base_columns = ['Home', 'Cinema', 'Method', 'Specific'] if 'Specific' in agg_df.columns else ['Home', 'Cinema', 'Method']
        agg_columns = base_columns + transport_types + ['TOTAL']
        
        # Ensure all required columns exist
        for col in base_columns:
            if col not in agg_df.columns:
                logger.warning(f"Column '{col}' not found in results DataFrame, creating empty column")
                agg_df[col] = ""
        
        # Create the final aggregate DataFrame
        final_agg_df = agg_df[agg_columns].copy()
        
        # Round numeric columns to 2 decimal places
        for col in transport_types + ['TOTAL']:
            final_agg_df[col] = final_agg_df[col].round(2)
        
        # Save to CSV
        logger.info(f"Saving aggregate data to {output_file}")
        final_agg_df.to_csv(output_file, index=False)
        logger.info(f"Aggregate data saved to {output_file}")
        
        # Display sample of output data
        logger.info(f"Sample aggregate output: \n{final_agg_df.head()}")
        
    except Exception as e:
        logger.error(f"Error generating aggregate data: {e}")
        raise

def test_api_connection():
    """Test the API connection with a simple request"""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logger.error("Google Maps API key not found. Please set GOOGLE_MAPS_API_KEY in .env file")
        return False
    
    logger.info("Testing API connection...")
    
    # Test known addresses that should definitely work
    origin = "SW1A 1AA, UK"  # Buckingham Palace postcode
    destination = "WC2N 5DU, UK"  # Trafalgar Square postcode
    
    # Test if we can geocode these addresses
    if not test_geocoding(origin) or not test_geocoding(destination):
        logger.error("Geocoding test failed. Please check your API key and make sure Geocoding API is enabled")
        return False
    
    # Now test the Routes API
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "routes,routes.legs,routes.legs.steps,routes.legs.steps.distanceMeters"
    }
    
    data = {
        "origin": {
            "address": origin
        },
        "destination": {
            "address": destination
        },
        "travelMode": "DRIVE",
        "routingPreference": "ROUTING_PREFERENCE_UNSPECIFIED",
        "languageCode": "en-GB"
    }
    
    try:
        logger.info("Sending test request to Google Routes API...")
        logger.info(f"Request body: {json.dumps(data, indent=2)}")
        
        response = requests.post(url, headers=headers, json=data)
        logger.info(f"Test response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Save response to a file for inspection
            with open("test_response.json", "w") as f:
                json.dump(result, f, indent=2)
            
            logger.info("API connection successful!")
            logger.info("Test response saved to test_response.json")
            
            if 'routes' in result and result['routes']:
                distance = result['routes'][0].get('distanceMeters', 'N/A')
                logger.info(f"Route found! Distance: {distance} meters")
                logger.info("Routes API is working correctly!")
                return True
            else:
                logger.warning("No routes found in the test response")
                return False
        else:
            logger.error(f"API connection failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"API connection test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting Google Routes API Distance Calculator")
    
    # First test API connection
    if test_api_connection():
        # If API test was successful, process the CSV
        process_csv()
    else:
        logger.error("API connection test failed. Please check your API key, enable necessary APIs in Google Cloud Console, and ensure billing is set up.")
        logger.error("Required APIs: Directions API, Geocoding API, and Routes API")