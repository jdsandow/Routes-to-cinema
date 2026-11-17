# Calculating the distances to cinema locations

You need a Google Maps API key to run this code. Add the key to an .env file in the root directory of the project. The .env file should look like this:

```bash
KEY="your_key_here"
```

## Notes
Where you do not know the preferred method of transit, but assume it is via public transport, use "TRANSIT_TRAVEL_MODE_UNSPECIFIED".
The Elizabeth line is classified as "Heavy_rail" by Google.

### Documentation references for travel mode/preferences/types
Route travel mode
https://developers.google.com/maps/documentation/routes/reference/rest/v2/RouteTravelMode

Transit preferences
https://developers.google.com/maps/documentation/routes/reference/rest/v2/TransitPreferences

Transit output types
https://googleapis.dev/dotnet/Google.Maps.Routing.V2/latest/api/Google.Maps.Routing.V2.TransitVehicle.Types.TransitVehicleType.html