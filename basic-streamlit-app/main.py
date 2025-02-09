import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/penguins.csv")
st.title(f"Exploring the Palmer Penguins Dataset")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Summary", "Exploring the Dataset by Species", "Exploring the Dataset by Island", "Exploring the Dataset by Weight", "Exploring the Dataset by Bills and Flippers"])

with tab1:
    st.header("Summary of the Dataset")
    # Data frame of entire dataset
    st.write("This app uses the dataset Palmer's penguins. It observes 344 individual penguins and identifies their species and the island on which they live. It records their bill length, bill depth, flipper length, and body mass. This app allows users to filter the dataset through the parameters previously stated. Users interact with selection boxes, sliders, check boxes, and buttons to filter the data. Formatted graphs and data frames allow the users to then analyze the data.")
    st.write("Here's the dataset loaded from a CSV file:")
    st.dataframe(df)
    total_count = df.shape[0]
    st.write("Here's the total number of penguins: ", total_count)
    st.write("There are three different species types:", df["species"].unique())
    st.write("There are three different islands:", df["island"].unique())



# Exploring the dataset by species
with tab2:
    # Data frame of a specific species
    st.header("Exploring the Dataset by Species")
    species = st.selectbox("Select a species", df["species"].unique())
    species_df = df[df["species"] == species]
    st.write("Here's the dataset filtered by species:")
    st.dataframe(species_df)

    # Average statistics of a specific species
    st.write(f"Here are some statistics for the {species} species:")
    avg_species_df = pd.DataFrame({
        f'Mean Statistics of {species} species': ['Bill Length (mm)', 'Bill Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)'],
        'Average Values': [species_df["bill_length_mm"].mean(), species_df["bill_depth_mm"].mean(), species_df["flipper_length_mm"].mean(), species_df["body_mass_g"].mean()]
    })
    st.dataframe(avg_species_df)



# Exploring the dataset by island
with tab3:
    # Data frame of a specific island
    st.header("Exploring the Dataset by Island")
    # Placing the buttons alongside each other
    col1, col2, col3 = st.columns(3)
    with col1:
        torgersen_button = st.button("Torgersen Island")
    with col2:
        biscoe_button = st.button("Biscoe Island")
    with col3:
        dream_button = st.button("Dream Island")
    # Selecting the island based on the button clicked
    if torgersen_button:
        island = "Torgersen"
    elif biscoe_button:
        island = "Biscoe"
    elif dream_button:
        island = "Dream"
    else:
        island = "Torgersen"
    island_df = df[df["island"] == island]

    # Bar plot of the number of each species on a specific island
    st.write(f"Here is a bar plot of the number of each species on {island} island:")
    # Including all species even when the count is 0
    all_species = pd.DataFrame(df["species"].unique(), columns=["species"])
    species_counts = island_df["species"].value_counts().reset_index()
    species_counts.columns = ["species", "count"]
    species_counts = all_species.merge(species_counts, on="species", how="left").fillna(0)
    # Plotting the bar plot
    sns.barplot(x="species", y="count", data=species_counts)
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.title(f'Number of Each Species on {island} Island')
    st.pyplot()

    # Statistics of a specific island
    st.write(f"Here are some statistics for {island} island:")
    avg_island_df = pd.DataFrame({
        f'Mean Statistics of {island} island': ['Bill Length (mm)', 'Bill Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)'],
        'Average Values': [island_df["bill_length_mm"].mean(), island_df["bill_depth_mm"].mean(), island_df["flipper_length_mm"].mean(), island_df["body_mass_g"].mean()]
    })
    st.dataframe(avg_island_df)


# Exploring the dataset by weight
with tab4:
    st.header("Exploring the Dataset by Weight")
    # Creating a histogram of the body mass of penguins by species using checkboxes
    # Creating separate data frames for each species
    adelie_df = df[df["species"] == "Adelie"]
    chinstrap_df = df[df["species"] == "Chinstrap"]
    gentoo_df = df[df["species"] == "Gentoo"]
    # Plotting the histogram
    st.write("Here is a histogram of the body mass of penguins by species:")
    # Creating checkboxes for each species
    adelie_checkbox = st.checkbox("Adelie", value=False)
    chinstrap_checkbox = st.checkbox("Chinstrap", value=False)
    gentoo_checkbox = st.checkbox("Gentoo", value=False)
    if adelie_checkbox:
        sns.histplot(data=adelie_df["body_mass_g"], label="Adelie", color = 'r')
    if chinstrap_checkbox:
        sns.histplot(data=chinstrap_df["body_mass_g"], label="Chinstrap", color = 'b')
    if gentoo_checkbox:
        sns.histplot(data=gentoo_df["body_mass_g"], label="Gentoo", color = 'g')
    plt.xlabel('Body Mass (g)')
    plt.ylabel('Count')
    plt.title('Body Mass of Penguins by Species')
    plt.legend()
    st.pyplot()

    # Use a slider to select a weight
    st.subheader("Selecting a Weight to Filter the Data")
    weight = st.slider("Select a weight (g)", df["body_mass_g"].min(), df["body_mass_g"].max())
    # Filter the data frame based on the selected weight
    # Display the number of penguins that weigh more than the selected weight by species
    weight_df = df[df["body_mass_g"] > weight]
    # Creating columns for species and island
    col1_2, col2_2 = st.columns(2)
    with col1_2:
        species_weight_counts = weight_df["species"].value_counts()
        species_weight_count_df = pd.DataFrame({
            'Species': species_weight_counts.index,
            'Count': species_weight_counts.values
        })
        st.write(f"Number of penguins that weigh more than {weight} grams by species:")
        st.dataframe(species_weight_count_df)
    with col2_2:
        island_weight_counts = weight_df["island"].value_counts()
        island_weight_count_df = pd.DataFrame({
            'Island': island_weight_counts.index,
            'Count': island_weight_counts.values
        })
        st.write(f"Number of penguins that weigh more than {weight} grams by island:")
        st.dataframe(island_weight_count_df)


# Exploring the dataset by bills and flippers
with tab5:
    st.header("Exploring the Dataset by Bills and Flippers")
    species_selected = st.multiselect("Select the species to display", df["species"].unique())
    col1_3, col2_3 = st.columns(2)
    with col1_3:
        # Scatter plot of bill length and bill depth colored by species
        st.write("Here is a scatter plot of bill length and bill depth colored by species:")
        sns.scatterplot(x="bill_length_mm", y="bill_depth_mm", hue="species", data=df[df["species"].isin(species_selected)])
        plt.xlabel('Bill Length (mm)')
        plt.ylabel('Bill Depth (mm)')
        plt.title('Bill Length vs. Bill Depth')
        plt.legend(title='Species')
        st.pyplot()

    with col2_3:
        # Bar plot of flipper length by island and species
        st.write("Here is a bar plot of flipper length by island and species:")
        sns.barplot(x="island", y="flipper_length_mm", hue="species", data=df[df["species"].isin(species_selected)])
        plt.xlabel('Island')
        plt.ylabel('Flipper Length (mm)')
        plt.title('Flipper Length by Island and Species')
        plt.legend(title='Species')
        st.pyplot()

    # Plotting bills and flippers against body mass
    st.subheader("Plotting Bills and Flippers Against Body Mass")
    st.write("Select the characteristic to plot against body mass:")
    characteristic = st.selectbox("Select a characteristic", df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]].columns)
    sns.scatterplot(x=characteristic, y="body_mass_g", hue="species", data=df)
    plt.xlabel(characteristic)
    plt.ylabel('Body Mass (g)')
    plt.title(f'Body Mass vs. {characteristic} by Species')
    plt.legend(title='Species')
    st.pyplot()