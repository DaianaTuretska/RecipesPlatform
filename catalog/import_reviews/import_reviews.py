#!/usr/bin/env python3
"""
import_comments_semantic.py

A standalone script (using classes) that:
  1) Loads a CSV of recipe‐comments,
  2) Strips HTML/newlines from both the CSV comments and your existing Recipe fields,
  3) Encodes each “combined recipe text” and each “comment text” using a pretrained
     Sentence-BERT model to get dense embeddings,
  4) Computes cosine‐similarity between each comment embedding and each recipe embedding,
  5) Creates a new Review object linking that comment to its best‐match Recipe,
     provided the similarity is above a given threshold.

Now using semantic embeddings (Sentence-BERT) rather than TF-IDF so that similarity
scores (e.g. 0.40, 0.60) are more meaningful than typical TF-IDF values around 0.06.

Instructions:
  • Place this file in the same directory as your manage.py.
  • Edit the `DJANGO_SETTINGS_MODULE` line below to your actual settings module.
  • Adjust CSV_PATH and MIN_SCORE at the bottom as desired.
  • Install dependencies if you haven’t:
      pip install pandas numpy beautifulsoup4 sentence-transformers django scikit-learn

Usage:
  python import_comments_semantic.py
"""

import os
import re
import sys
import random

# ────────────────────────────────────────────────────────────────────────────────
# STEP ZERO: Point to your Django settings before importing django or any models.
#           Replace "myproject.settings" with your actual settings path.
# ────────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "website.settings")
# ────────────────────────────────────────────────────────────────────────────────

import django

django.setup()

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from django.db import transaction

from catalog.models import Recipe, Review
from users.models import User


class TextPreprocessor:
    @staticmethod
    def strip_html_and_newlines(html: str) -> str:
        """
        1) Remove HTML tags via BeautifulSoup.
        2) Replace newline/carriage‐return (\n, \r) with spaces.
        3) Collapse multiple whitespace into exactly one space.
        """
        if not html:
            return ""

        # a) Strip HTML tags
        text_only = BeautifulSoup(html, "html.parser").get_text(separator=" ")

        # b) Replace newlines/carriage‐returns with a space
        text_no_newline = text_only.replace("\n", " ").replace("\r", " ")

        # c) Collapse multiple whitespace into a single space
        text_clean = re.sub(r"\s+", " ", text_no_newline).strip()

        return text_clean

    @staticmethod
    def collapse_whitespace(text: str) -> str:
        """
        Replace any stray newlines in plain text fields with spaces,
        then collapse runs of whitespace into a single space.
        """
        if not text:
            return ""
        no_newline = text.replace("\n", " ").replace("\r", " ")
        return re.sub(r"\s+", " ", no_newline).strip()


class CommentToRecipeImporter:
    def __init__(self, csv_path: str, min_score: float = 0.40):
        """
        Args:
          csv_path: Full path to your "Recipe Reviews and User Feedback" CSV.
          min_score: Minimum cosine‐similarity required to attach a comment.
                     Sentence-BERT similarities for related texts often run 0.30–0.80,
                     so a threshold of 0.40 or 0.45 is reasonable.
        """
        self.csv_path = csv_path
        self.min_score = min_score

        # These will be filled in during .run():
        self.df_recipes = None
        self.df_comments = None
        self.recipe_embeddings = None
        self.comment_embeddings = None
        self.best_match_df = None

    def load_and_clean_recipes(self):
        """
        1) Query all Recipe rows from Django into a DataFrame.
        2) Strip HTML + newlines from the rich‐text fields (ingredients, description, cooking_method).
        3) Collapse whitespace in title, cuisine, category.
        4) Build a “combined_text” column for each recipe.
        """
        print("=== Loading recipes from database ===")
        qs = Recipe.objects.all().values(
            "id",
            "title",
            "ingredients",
            "description",
            "cooking_method",
            "cuisine",
            "category",
        )
        self.df_recipes = pd.DataFrame.from_records(
            qs,
            columns=[
                "id",
                "title",
                "ingredients",
                "description",
                "cooking_method",
                "cuisine",
                "category",
            ],
        )

        print(f"  → Raw df_recipes shape: {self.df_recipes.shape}")
        print("  → Raw df_recipes.head():")
        print(self.df_recipes.head(), "\n")

        # If any field is None/NaN, fill with empty string, then strip HTML/newlines
        for col in ["ingredients", "description", "cooking_method"]:
            self.df_recipes[col] = (
                self.df_recipes[col]
                .fillna("")
                .apply(TextPreprocessor.strip_html_and_newlines)
            )

        # Collapse whitespace in title/cuisine/category
        for col in ["title", "cuisine", "category"]:
            self.df_recipes[col] = (
                self.df_recipes[col]
                .fillna("")
                .apply(TextPreprocessor.collapse_whitespace)
            )

        # Build combined_text = title + ingredients + description + cooking_method + cuisine + category
        self.df_recipes["combined_text"] = (
            self.df_recipes["title"]
            + " "
            + self.df_recipes["ingredients"]
            + " "
            + self.df_recipes["description"]
            + " "
            + self.df_recipes["cooking_method"]
            + " "
            + self.df_recipes["cuisine"]
            + " "
            + self.df_recipes["category"]
        )

        print(f"  → Cleaned df_recipes shape: {self.df_recipes.shape}")
        print("  → Cleaned df_recipes.head():")
        print(self.df_recipes.head(), "\n")

    def load_and_clean_comments(self):
        """
        1) Read the CSV with pandas.
        2) Keep columns: recipe_name (for reference), comment_id (unique), comment_text (the raw text), user_name.
        3) Collapse whitespace / drop newlines from comment_text.
        """
        print("=== Loading comments from CSV ===")
        print(f"CSV path: {self.csv_path}")
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"ERROR: Could not read CSV at '{self.csv_path}':\n  {e}")
            sys.exit(1)

        print(f"  → Raw CSV shape: {df.shape}")
        print("  → Raw CSV head:")
        print(df.head(), "\n")

        # Ensure required columns are present
        required = {"recipe_name", "comment_id", "text", "user_name"}
        missing = required - set(df.columns)
        if missing:
            print(f"ERROR: CSV is missing required columns: {missing}")
            sys.exit(1)

        # Rename “text” → “comment_text”, keep only needed columns
        self.df_comments = df.rename(
            columns={"recipe_name": "csv_recipe_name", "text": "comment_text"}
        )[["csv_recipe_name", "comment_id", "comment_text", "user_name"]]

        # Clean comment_text: replace newlines with spaces, collapse whitespace
        self.df_comments["comment_text"] = (
            self.df_comments["comment_text"]
            .fillna("")
            .apply(TextPreprocessor.collapse_whitespace)
        )

        print(f"  → Cleaned df_comments shape: {self.df_comments.shape}")
        print("  → Cleaned df_comments.head():")
        print(self.df_comments.head(), "\n")

    def encode_recipe_embeddings(self):
        """
        1) Use a SentenceTransformer model to encode each recipe’s combined_text into a dense vector.
        2) Normalize embeddings so that cosine similarity = dot product.
        """
        print("=== Encoding recipe texts with Sentence-BERT ===")
        model = SentenceTransformer("all-MiniLM-L6-v2")  # small+fast model

        recipe_texts = self.df_recipes["combined_text"].tolist()
        print("  → Computing recipe embeddings...")
        emb = model.encode(recipe_texts, convert_to_numpy=True, show_progress_bar=True)

        # Normalize to unit length
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.recipe_embeddings = emb
        print(f"  → recipe_embeddings shape: {emb.shape}\n")

    def encode_comment_embeddings(self):
        """
        1) Use the same SentenceTransformer model to encode each comment’s text.
        2) Normalize embeddings.
        """
        print("=== Encoding comment texts with Sentence-BERT ===")
        model = SentenceTransformer("all-MiniLM-L6-v2")  # same model

        comment_texts = self.df_comments["comment_text"].tolist()
        print("  → Computing comment embeddings...")
        emb = model.encode(comment_texts, convert_to_numpy=True, show_progress_bar=True)

        # Normalize to unit length
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.comment_embeddings = emb
        print(f"  → comment_embeddings shape: {emb.shape}\n")

    def compute_semantic_similarity_and_match(self):
        """
        1) Compute cosine‐similarity between each comment embedding and each recipe embedding.
        2) Find the highest‐scoring recipe for each comment.
        3) Build best_match_df with (comment_id, matched_recipe_pk, similarity_score).
        """
        print("=== Computing semantic cosine‐similarity & matching ===")
        # comment_embeddings: shape (num_comments, dim)
        # recipe_embeddings:  shape (num_recipes, dim)
        sim_matrix = cosine_similarity(self.comment_embeddings, self.recipe_embeddings)
        print(f"  → similarity matrix shape: {sim_matrix.shape}")

        num_comments = len(self.df_comments)
        # For each comment (row i), pick best recipe index
        best_recipe_idx = np.argmax(sim_matrix, axis=1)
        best_scores = sim_matrix[np.arange(num_comments), best_recipe_idx]
        matched_recipe_pks = self.df_recipes.loc[best_recipe_idx, "id"].values

        self.best_match_df = pd.DataFrame(
            {
                "comment_id": self.df_comments["comment_id"],
                "csv_recipe_name": self.df_comments["csv_recipe_name"],
                "user_name": self.df_comments["user_name"],
                "matched_recipe_pk": matched_recipe_pks,
                "similarity_score": best_scores,
            }
        )

        print(f"  → best_match_df shape: {self.best_match_df.shape}")
        print("  → best_match_df.head():")
        print(self.best_match_df.head(), "\n")

    def create_reviews_in_db(self):
        """
        Create Review objects in bulk rather than one‐by‐one.

        1) Create or fetch a fallback “import_user” to own imported reviews.
        2) Pre‐collect existing “[Imported:<comment_id>]” markers to avoid duplicates.
        3) For each row in self.best_match_df:
             a) If similarity_score >= self.min_score:
                i)  Check for existing marker to skip duplicates.
                ii) Build a Review instance (in memory) with comment prefixed.
                    Assign a random existing user (or fallback to import_user).
        4) Call bulk_create(...) on the list of new Review instances.
        5) Report how many were created vs. skipped.
        """
        print(
            "=== Creating Review objects via bulk_create (random‐user assignment) ==="
        )

        # 2) Fetch all existing review‐markers (so we don’t create duplicates).
        #    We assume imported reviews always start with “[Imported:<comment_id>] ”.
        #    Build a set of those markers already in the DB:
        existing_markers = set(
            Review.objects.filter(comment__startswith="[Imported:").values_list(
                "comment", flat=True
            )
        )
        # Extract just the “[Imported:<comment_id>]” portion from each existing comment:
        existing_comment_ids = set()
        for full_comment in existing_markers:
            # full_comment looks like "[Imported:sp_... ] actual text…"
            # We split on the first space to isolate the marker itself:
            marker = full_comment.split(" ", 1)[0]  # e.g. "[Imported:sp_...]"
            comment_id = marker.strip("[]").split("Imported:")[1]
            existing_comment_ids.add(comment_id)

        # 3) Pre‐load all user IDs once (for random selection)
        all_user_ids = list(User.objects.values_list("id", flat=True))

        new_reviews = []  # list to hold Review instances (unsaved)

        skipped_low_score = 0
        skipped_exists = 0

        # 4) Iterate through best_match_df and build Review objects in memory
        for _, row in self.best_match_df.iterrows():
            comment_id = row["comment_id"]
            matched_pk = row["matched_recipe_pk"]
            score = row["similarity_score"]
            orig_username = row["user_name"]

            # a) Skip if below similarity threshold
            if score < self.min_score:
                skipped_low_score += 1
                continue

            # b) Skip if this comment_id was already imported
            if comment_id in existing_comment_ids:
                skipped_exists += 1
                continue

            # c) Fetch the matched Recipe (skip if deleted)
            try:
                recipe_obj = Recipe.objects.get(pk=matched_pk)
            except Recipe.DoesNotExist:
                continue

            # d) Decide which User to assign:
            #    Pick a random existing user, fallback to import_user if no users exist.
            if all_user_ids:
                chosen_id = random.choice(all_user_ids)
                assigned_user = User.objects.get(pk=chosen_id)

            # e) Build “[Imported:<comment_id>] actual comment_text”
            original_text = self.df_comments.loc[
                self.df_comments["comment_id"] == comment_id, "comment_text"
            ].values[0]
            comment_to_save = original_text

            # f) Create a new Review instance in memory (not yet saved)
            new_reviews.append(
                Review(
                    user=assigned_user,
                    recipe=recipe_obj,
                    status="active",
                    comment=comment_to_save,
                )
            )

            # Mark this comment_id as now “existing” so we don’t duplicate
            existing_comment_ids.add(comment_id)

        # 5) Bulk‐insert all Review instances at once
        with transaction.atomic():
            Review.objects.bulk_create(new_reviews)

        print(f"  → {len(new_reviews)} Review(s) created via bulk_create.")
        print(f"  → {skipped_low_score} skipped (similarity < {self.min_score}).")
        print(f"  → {skipped_exists} skipped (already existed).")
        print("All done.\n")

    def run(self, create_reviews: bool = True) -> None:
        """
        Execute the full pipeline:
          1) load & clean recipes,
          2) load & clean comments,
          3) encode embeddings,
          4) compute semantic similarity & match,
          5) (optionally) create Review objects.
        """
        self.load_and_clean_recipes()
        self.load_and_clean_comments()
        self.encode_recipe_embeddings()
        self.encode_comment_embeddings()
        self.compute_semantic_similarity_and_match()

        if create_reviews:
            self.create_reviews_in_db()


if __name__ == "__main__":
    # ────────────────────────────────────────────────────────────────────────────────
    # CONFIGURE THESE VARIABLES as needed:

    # 1) Locate the CSV relative to this script’s directory (or use an absolute path).
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(
        current_dir,
        "assets",
        "Recipe Reviews and User Feedback Dataset.csv",
    )

    # 2) Set your minimum Sentence-BERT cosine‐similarity threshold.
    #    Related comments often score 0.30–0.80. We choose 0.40 by default.
    MIN_SCORE = 0.40
    # ────────────────────────────────────────────────────────────────────────────────

    importer = CommentToRecipeImporter(csv_path=csv_path, min_score=MIN_SCORE)

    # If you only want to see DataFrame heads and skip writing Reviews:
    #    importer.run(create_reviews=False)
    #
    # To run the full pipeline (including creating Review rows), use:
    importer.run(create_reviews=True)
