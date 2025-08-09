import streamlit as st
import pandas as pd
import openai
import json
import os

# --- App Configuration ---
st.set_page_config(layout="wide")

# --- Language Selection ---
def load_translation(language):
    path = os.path.join("locales", f"{language}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

languages = {"简体中文": "zh", "English": "en", "Deutsch": "de"}
selected_language_name = st.sidebar.selectbox("Language", list(languages.keys()))
lang_code = languages[selected_language_name]
t = load_translation(lang_code)

# --- Sidebar for AI Configuration ---
st.sidebar.header(t["sidebar_ai_config_header"])
api_key = st.sidebar.text_input(t["sidebar_api_key_prompt"], type="password")
base_url = st.sidebar.text_input(t["sidebar_base_url_prompt"], "https://api.openai.com/v1")
model_name = st.sidebar.text_input(t["sidebar_model_name_prompt"], "gpt-4")

st.sidebar.header(t["sidebar_prompt_template_header"])
default_prompt = """下面是一个Markdown格式的表格，请你分析这个表格并总结其中的关键信息和趋势。

{{markdown_table}}"""
prompt_template = st.sidebar.text_area(t["sidebar_edit_prompt"], default_prompt, height=300)

# --- Main App ---
st.title(t["app_title"])

uploaded_file = st.file_uploader(t["upload_prompt"], type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.header(t["original_data_header"])
        st.dataframe(df)

        st.header(t["select_cols_filter_rows_header"])
        selected_columns = st.multiselect(t["select_columns_prompt"], df.columns.tolist(), default=df.columns.tolist())
        
        # Row Filters
        st.write(t["row_filter_prompt"])
        filtered_df = df.copy()
        filter_cols = st.columns(len(selected_columns))
        for i, col in enumerate(selected_columns):
            with filter_cols[i]:
                filter_value = st.text_input(t["filter_column_prompt"].format(col=col), key=f"filter_{col}")
                if filter_value:
                    terms = [term.strip() for term in filter_value.split(',') if term.strip()]
                    include_terms = [t for t in terms if not t.startswith('-')]
                    exclude_terms = [t[1:] for t in terms if t.startswith('-') and len(t) > 1]

                    if exclude_terms:
                        filtered_df = filtered_df[~filtered_df[col].astype(str).str.contains('|'.join(exclude_terms), case=False, na=False)]
                    
                    if include_terms:
                        filtered_df = filtered_df[filtered_df[col].astype(str).str.contains('|'.join(include_terms), case=False, na=False)]

        if selected_columns:
            sub_df = filtered_df[selected_columns]

            st.header(t["sub_table_preview_header"])
            st.dataframe(sub_df)

            st.header(t["rename_columns_header"])
            new_column_names = {}
            cols = st.columns(len(sub_df.columns))
            for i, col in enumerate(sub_df.columns):
                new_name = cols[i].text_input(t["rename_column_prompt"].format(col=col), value=col, key=f"col_{i}")
                new_column_names[col] = new_name

            renamed_df = sub_df.rename(columns=new_column_names)

            st.header(t["final_markdown_header"])
            markdown_table = renamed_df.to_markdown(index=False)
            st.code(markdown_table, language='markdown')

            # --- AI Analysis Section ---
            st.header(t["ai_analysis_header"])
            if st.button(t["start_analysis_button"]):
                if not api_key:
                    st.warning(t["warning_no_api_key"])
                elif '{{markdown_table}}' not in prompt_template:
                    st.warning(t["warning_no_placeholder"])
                else:
                    try:
                        client = openai.OpenAI(api_key=api_key, base_url=base_url)
                        final_prompt = prompt_template.replace("{{markdown_table}}", markdown_table)
                        
                        with st.spinner(t["spinner_ai_analyzing"]):
                            stream = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": final_prompt}],
                                stream=True,
                            )
                            response = st.write_stream(stream)

                    except Exception as e:
                        st.error(t["error_calling_ai"].format(e=e))

    except Exception as e:
        st.error(t["error_processing_file"].format(e=e))
