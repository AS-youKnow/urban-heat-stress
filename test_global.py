from config import WORLD_REGIONS, get_synthetic_dataframe

cities = ['Delhi, India', 'London, UK', 'Moscow, Russia', 'Lagos, Nigeria', 'Sydney, Australia', 'Riyadh, Saudi Arabia']
print(f"{'City':<32} {'Mean LST':>10} {'Min':>8} {'Max':>8} {'Lat Centre':>12}")
print("-" * 75)
for city in cities:
    bbox = WORLD_REGIONS[city]
    df = get_synthetic_dataframe(bbox=bbox, region_name=city)
    lat_c = (bbox[1] + bbox[3]) / 2
    print(f"{city:<32} {df['LST_Celsius'].mean():>9.1f}C {df['LST_Celsius'].min():>7.1f} {df['LST_Celsius'].max():>7.1f} {lat_c:>11.1f}deg")

print("\nAll cities OK!")
