# Calculating the distances to cinema locations

You need a Google Maps API key to run this code. Add the key to an .env file in the root directory of the project. The .env file should look like this:

```bash
GOOGLE_MAPS_API_KEY="your_key_here"
```

The travel-to-cinema.csv file gives the expected structure of the data, and includes sample data.

If a route uses public transport, put "TRANSIT" in the 'Method' column. If you know what kind of public transport is used, add this into the 'Specific' column.

Allowed forms of public transport are:
BUS
SUBWAY (e.g., the London Underground)
TRAIN
LIGHT_RAIL (includes trams)
RAIL (this is a catch-all for any rail transport)

(I need to check exactly how various modes of rail are handled (e.g., DLR, Overground, etc).)

Outputs: 
A CSV file with the same structure as the input file, but with additional columns for every step of the route, the method used, and the distance.
A CSV file with the same structure as the input file, but with additional columns that aggregate the distance taken for each method.

Distance is in kilometers.