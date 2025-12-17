import sys
import os

# Ensure import path
sys.path.append(os.getcwd())

from data_enrichment.tmdb_client import search_movie, search_movies_list

def debug():
    print("🔎 Debugging TMDB Logic for 'Sikandar'")
    
    # 1. Test List Search (What the user sees)
    print("\n1. Search List ('Sikandar'):")
    results = search_movies_list("Sikandar")
    for m in results:
        year = m.get('release_date', 'N/A')[:4]
        print(f"   found: {m['title']} ({year}) -> ID: {m['id']}")

    # 2. Test Specific Year Search (What lookup_movie uses)
    print("\n2. Specific Lookup ('Sikandar', year=2025):")
    match_2025 = search_movie("Sikandar", year=2025)
    if match_2025:
        print(f"   ✅ FOUND: {match_2025['title']} ({match_2025.get('release_date')})")
    else:
        print("   ❌ NOT FOUND with year=2025")
        
    # 3. Test Specific Year Search (2024? just in case)
    print("\n3. Specific Lookup ('Sikandar', year=2024):")
    match_2024 = search_movie("Sikandar", year=2024)
    if match_2024:
        print(f"   ✅ FOUND: {match_2024['title']} ({match_2024.get('release_date')})")
    else:
        print("   ❌ NOT FOUND with year=2024")

if __name__ == "__main__":
    debug()
