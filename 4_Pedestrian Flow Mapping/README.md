# Module 4 - Pedestrian flow mapping

Two parts share the reconstructed pedestrian network of Module 5:

| Part | Folder | Question answered | Core |
|---|---|---|---|
| a | `a_OD Coolest-Route Algorithm (route cost calculation)/` | How much of every street segment is shaded at 14:00, what does a metre of sunlit street cost a walker relative to a shaded one, and which route is the coolest between two points? | per-edge shade fraction from the 14:00 shadow raster of Module 1; edge cost `omega = length x ((1 - shade) + lambda)`; station-anchored OD demand with distance decay; Dijkstra routing; lambda / detour sensitivity; the ShadeWalk web tool |
| b | `b_Pedestrian OD Flow Assignment (building gravity and huff model)/` | How is the 14:00 transit ridership (bus stops, MRT / LRT exits) distributed to the buildings weighted by floor area and hourly occupancy, and how much flow does each street segment carry under the shortest-path and the coolest-path rule? | demand tables (station ridership, building weights, occupancy density); Huff destination choice with exponential distance decay (350 m), 400 m bus / 800 m rail catchments; shortest and coolest routing of every OD pair on the edge costs of part a; lambda / tau sensitivity |

Part a produces the routing graph with per-edge shade (`step4_4_edges_SG.gpkg`, `step4_4_nodes_SG.gpkg`) and the per-edge facility class that part b routes on, and generates the ShadeWalk web tool. Part b produces the station ridership layer (`station_hourly_ridership_v4.gpkg`: one record per real MRT / LRT exit point or bus stop, an interchange counted once), the hourly building weights
(`building_hourly_weight_gfa.gpkg`; gross floor area = `gfa_corr` of the released building dataset), the OSM pedestrian subset
(`pedestrian_network_filtered.gpkg`) used by Modules 2 and 5, and the per-edge flow products. Each part has its own README and `INPUT_DATA.md`.
