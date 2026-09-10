# Module 4 - Pedestrian flow mapping

Two complementary flow models run on the reconstructed pedestrian network of Module 5:

| Part | Folder | Question answered | Core |
|---|---|---|---|
| a | `a_OD Coolest-Route Algorithm (Route Cost Calculation)/` | Which route does a pedestrian take between a transit station and a building when sun exposure is penalised, and how much flow does each street segment carry under the shortest-path and coolest-path assumptions? | per-edge shade fraction from the 14:00 shadow raster of Module 1; edge cost `omega = length x ((1 - shade) + lambda)`; station-anchored OD demand with distance decay; Dijkstra routing; lambda / detour sensitivity; the ShadeWalk web tool |
| b | `b_Pedestrian OD Flow Assignment (Gravity and Huff model)/` | How is the hourly transit ridership (bus stops, MRT / LRT exits) distributed over the footpath network towards buildings weighted by floor area and hourly occupancy? | madina *patronage betweenness*: Huff-style destination competition with an exponential distance decay, 400 m bus / 800 m rail catchments, dual-pass ingress + egress |

Part b produces the station ridership layer (`station_hourly_ridership.gpkg`) and the hourly building weights
(`building_hourly_weight.gpkg`) that part a uses as origin and destination weights, and the OSM pedestrian subset
(`pedestrian_network_filtered.gpkg`) used by Modules 2 and 5. Each part has its own README and `INPUT_DATA.md`.
