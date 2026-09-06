import os
from pathlib import Path

import polars as pl
import typer
from anthropic import Anthropic
from pydantic import BaseModel, Field
from tqdm import tqdm

# Constant for when no common names are found
NO_COMMON_NAMES_PLACEHOLDER = ""


class SpeciesNameMapping(BaseModel):
    """A mapping from scientific name to common names."""

    scientific_name: str = Field(description="The scientific name of the species")
    common_names: list[str] = Field(
        description="All commonly used English common names (vernacular names) for this species, ordered from most to least common. Include at least 1-3 names if they exist. Do not leave empty."
    )


class CommonNamesResponse(BaseModel):
    """Response containing common names for multiple species."""

    species_mappings: list[SpeciesNameMapping] = Field(
        description="List of scientific name to common name mappings"
    )


COMMON_NAME_PROMPT = """You are a botanical expert. For each plant species listed below, provide ALL of its commonly used English common names (vernacular names). Include at least 1-3 common names for each species if they exist. Order the names from most commonly used to least commonly used.

For example:
- Acer rubrum might have: Red Maple, Swamp Maple, Soft Maple
- Taraxacum officinale might have: Dandelion, Common Dandelion, Blowball

Species list:
{species_list}

Provide the common names for each of these species."""

# Claude Haiku 4.5 pricing (per million tokens) - much cheaper than Sonnet!
INPUT_TOKEN_COST = 0.25  # $0.25 per million input tokens
OUTPUT_TOKEN_COST = 1.25  # $1.25 per million output tokens


def get_common_names_batch(
    scientific_names: list[str], client: Anthropic
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """
    Query the Anthropic API to get common names for a batch of species.

    Args:
        scientific_names: List of scientific names
        client: Anthropic client

    Returns:
        Tuple of (dictionary mapping scientific names to lists of common names, usage dict)
    """
    species_list = "\n".join(f"- {name}" for name in scientific_names)
    prompt = COMMON_NAME_PROMPT.format(species_list=species_list)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        tools=[
            {
                "name": "provide_common_names",
                "description": "Provide common names for the given plant species",
                "input_schema": CommonNamesResponse.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": "provide_common_names"},
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract the structured output from tool use
    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"), None
    )

    if not tool_use_block:
        raise ValueError("No tool use block found in response")

    # Parse with Pydantic
    common_names_response = CommonNamesResponse.model_validate(tool_use_block.input)

    # Convert to dictionary
    result = {
        mapping.scientific_name: mapping.common_names
        for mapping in common_names_response.species_mappings
    }

    # Extract usage information
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    return result, usage


def get_common_names(input_file: str, output_file: str, batch_size: int = 50):
    """
    Read species names from a TSV file and get all their common names.

    Args:
        input_file: Path to input TSV file with a 'name' column
        output_file: Path to output TSV file (common names will be semicolon-separated)
        batch_size: Number of species to query at once
    """
    # Read the input file and get unique names
    df = pl.read_csv(input_file, separator="\t")

    if "name" not in df.columns:
        print(f"Error: 'name' column not found in {input_file}")
        raise typer.Exit(1)

    unique_names = df["name"].unique().sort().to_list()

    print(f"Found {len(unique_names)} unique species names")

    # Initialize Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        raise typer.Exit(1)

    client = Anthropic(api_key=api_key)

    # Estimate cost before starting
    num_batches = (len(unique_names) + batch_size - 1) // batch_size

    # Estimate tokens per batch
    # Input: prompt template (~100 tokens) + schema (~500 tokens) + species names (~15 tokens each)
    estimated_input_per_batch = 600 + (batch_size * 15)
    # Output: structured response with ~40 tokens per species (scientific name + 2-4 common names)
    estimated_output_per_batch = batch_size * 40

    estimated_total_input = estimated_input_per_batch * num_batches
    estimated_total_output = estimated_output_per_batch * num_batches

    estimated_input_cost = (estimated_total_input / 1_000_000) * INPUT_TOKEN_COST
    estimated_output_cost = (estimated_total_output / 1_000_000) * OUTPUT_TOKEN_COST
    estimated_total_cost = estimated_input_cost + estimated_output_cost

    print(f"\nEstimated Cost:")
    print(f"  Number of batches: {num_batches}")
    print(f"  Estimated input tokens:  ~{estimated_total_input:,}")
    print(f"  Estimated output tokens: ~{estimated_total_output:,}")
    print(f"  Estimated total cost:    ~${estimated_total_cost:.4f}")

    # Query API in batches and write incrementally
    all_common_names = {}
    total_input_tokens = 0
    total_output_tokens = 0

    # Create/overwrite the output file with headers
    with open(output_file, "w") as f:
        f.write("scientific_name\tcommon_names\n")

    print(f"\nQuerying Anthropic API in batches of {batch_size}...")
    print(f"Writing results incrementally to {output_file}")

    with tqdm(
        total=len(unique_names), desc="Getting common names", unit="species"
    ) as pbar:
        for i in range(0, len(unique_names), batch_size):
            batch = unique_names[i : i + batch_size]

            try:
                batch_results, usage = get_common_names_batch(batch, client)
                all_common_names.update(batch_results)
                total_input_tokens += usage["input_tokens"]
                total_output_tokens += usage["output_tokens"]

                # Write batch results immediately
                with open(output_file, "a") as f:
                    for sci_name, common_names_list in batch_results.items():
                        common_names_str = (
                            "; ".join(common_names_list)
                            if common_names_list
                            else NO_COMMON_NAMES_PLACEHOLDER
                        )
                        f.write(f"{sci_name}\t{common_names_str}\n")

                pbar.update(len(batch))
            except Exception as e:
                print(f"\nError processing batch starting at index {i}: {e}")
                # Continue with remaining batches
                pbar.update(len(batch))

    # Read back the file for summary display
    output_df = pl.read_csv(output_file, separator="\t")

    print(f"\nWrote {len(output_df)} species mappings to {output_file}")
    print(
        f"Successfully retrieved common names for {len(all_common_names)}/{len(unique_names)} species"
    )

    # Show a sample
    if len(output_df) > 0:
        print("\nSample entries:")
        for row in output_df.head(3).iter_rows(named=True):
            print(f"  {row['scientific_name']}: {row['common_names']}")

    # Calculate and display cost
    input_cost = (total_input_tokens / 1_000_000) * INPUT_TOKEN_COST
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_TOKEN_COST
    total_cost = input_cost + output_cost

    print(f"\n{'='*60}")
    print(f"API Usage Summary:")
    print(
        f"  Input tokens:  {total_input_tokens:,} (estimated: ~{estimated_total_input:,})"
    )
    print(
        f"  Output tokens: {total_output_tokens:,} (estimated: ~{estimated_total_output:,})"
    )
    print(f"  Total tokens:  {total_input_tokens + total_output_tokens:,}")
    print(f"\nCost Breakdown:")
    print(f"  Input cost:  ${input_cost:.4f}")
    print(f"  Output cost: ${output_cost:.4f}")
    print(f"  Total cost:  ${total_cost:.4f} (estimated: ~${estimated_total_cost:.4f})")
    print(f"{'='*60}")


app = typer.Typer()


@app.command()
def main(
    input_file: str = typer.Argument(
        ..., help="Path to the input TSV file (e.g., data/plant_pics.tsv)"
    ),
    output_file: str = typer.Option(
        "data/common_names.tsv", "--output", "-o", help="Path to the output TSV file"
    ),
    batch_size: int = typer.Option(
        50, "--batch-size", "-b", help="Number of species to query per API call"
    ),
):
    """
    Get all common names for plant species using the Anthropic API.

    Reads scientific names from the 'name' column of the input file,
    queries the Anthropic API for all common names (ordered from most to least common),
    and writes the results to an output TSV file with common names separated by semicolons.
    """
    get_common_names(input_file, output_file, batch_size)


if __name__ == "__main__":
    app()
