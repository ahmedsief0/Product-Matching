import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

import re
unwanted_words = {
    "عرض", "س ج", "تخفيض", "مجانا", "حصري", "شحن", "مجانى", "السعر",
    "خصم", "هديه", "قديم", "جديد", "سعر", "العامريه", "شريط",
    "ج", "اقراص", "حقن", "س", "العامريه", "ب", "نقط"
}

def clean_arabic_text(text):
    if isinstance(text, str):
        # توحيد الأحرف العربية
        text = re.sub(r'[إأآا]', 'ا', text)
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ؤ', 'و', text)
        text = re.sub(r'ئ', 'ي', text)
        text = re.sub(r'ء', '', text)
        text = re.sub(r'ة', 'ه', text)

        # إزالة التشكيل
        arabic_diacritics = re.compile(r'''
            ّ    | # الشدة
            َ    | # الفتحة
            ً    | # تنوين الفتح
            ُ    | # الضمة
            ٌ    | # تنوين الضم
            ِ    | # الكسرة
            ٍ    | # تنوين الكسر
            ْ    | # السكون
            ـ       # التطويل
        ''', re.VERBOSE)
        text = re.sub(arabic_diacritics, '', text)

        # إضافة مسافات بين الأرقام والحروف إذا لم تكن موجودة
        text = re.sub(r'(\D)(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)(\D)', r'\1 \2', text)

        # إزالة العلامات غير المرغوب فيها والمسافات الزائدة
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # تحويل النص إلى حروف صغيرة وتقسيمه إلى كلمات
        words = text.strip().lower().split()

        # قائمة لتجميع الكلمات المنظفة
        cleaned_words = []
        unwanted_all = unwanted_words.union({
            "جنيه", "ريال", "درهم", "دينار", "دولار", "يورو",
            "ج", "ج م", "جم", "ر س", "رس", "السعر", "ليره",
            "روبيه", "جنيه مصري"
        })

        skip_numbers = False
        for word in words:
            if skip_numbers:
                if not word.isdigit() and not re.match(r'\d+(\.\d+)?', word):
                    skip_numbers = False

            if word in unwanted_all:
                skip_numbers = True
                continue  # لا نضيف الكلمة غير المرغوب فيها
            elif skip_numbers and (word.isdigit() or re.match(r'\d+(\.\d+)?', word)):
                continue  # نتخطى الأرقام بعد الكلمة غير المرغوب فيها
            else:
                cleaned_words.append(word)

        cleaned_text = ' '.join(cleaned_words)
        return cleaned_text.strip()

    return ''
def load_data(file_path):
    """Load and preprocess data with price rounding."""
    xls = pd.ExcelFile(file_path)
    #xls2 = pd.ExcelFile(file_path)
    #dataset = pd.read_excel(xls2, "Dataset",
                            #usecols=['sku', 'seller_item_name', 'price'],
                            #dtype={'sku': str, 'price': float})
    master = pd.read_excel(xls, "Master File", 
                           usecols=['sku', 'product_name_ar', 'price'],
                           dtype={'sku': str, 'price': float})
    dataset = pd.read_excel(xls, "Dataset",
                            usecols=['sku', 'seller_item_name', 'price'],
                            dtype={'sku': str, 'price': float})
    
    # Clean and normalize text
    master['clean_name'] = master['product_name_ar'].apply(clean_arabic_text)
    dataset['clean_name'] = dataset['seller_item_name'].apply(clean_arabic_text)
    
    # Standardize prices to 2 decimal places
    master['price'] = master['price'].round(2)
    dataset['price'] = dataset['price'].round(2)
    
    return master, dataset

def match_products(master, dataset):
    """Combined matching: 75% price similarity and 25% text similarity."""
    print("Starting combined matching (75% price, 25% text)...")
    start_time = time.time()
    
    # Create TF-IDF vectors for text similarity
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    master_tfidf = vectorizer.fit_transform(master['clean_name'])
    dataset_tfidf = vectorizer.transform(dataset['clean_name'])
    
    # Calculate text similarity matrix
    text_similarity_matrix = cosine_similarity(dataset_tfidf, master_tfidf)
    
    # Precompute master prices as a numpy array
    master_prices = master['price'].values
    
    def compute_price_similarity(ds_price, master_price):
        """Compute price similarity between two prices."""
        if ds_price == 0 and master_price == 0:
            return 1
        elif ds_price == 0 or master_price == 0:
            return 0
        else:
            sim = 1 - abs(master_price - ds_price) / max(master_price, ds_price)
            return max(0, sim)
    
    matched_indices = []
    combined_scores = []
    text_scores = []
    price_scores = []
    
    for i in range(len(dataset)):
        ds_price = dataset.iloc[i]['price']
        # Compute price similarity vector for all master items for the current dataset item
        price_similarity_vector = np.array([compute_price_similarity(ds_price, mp) for mp in master_prices])
        
        # Combined score: 25% text similarity and 75% price similarity
        combined_vector = 0.25 * text_similarity_matrix[i] + 0.75 * price_similarity_vector
        
        best_match_idx = np.argmax(combined_vector)
        
        matched_indices.append(best_match_idx)
        combined_scores.append(combined_vector[best_match_idx])
        text_scores.append(text_similarity_matrix[i, best_match_idx])
        price_scores.append(price_similarity_vector[best_match_idx])
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Dataset_SKU': dataset['sku'],
        'Seller_Item': dataset['seller_item_name'],
        'Seller_Price': dataset['price'],
        'Matched_SKU': master.iloc[matched_indices]['sku'].values,
        'Master_Item': master.iloc[matched_indices]['product_name_ar'].values,
        'Master_Price': master.iloc[matched_indices]['price'].values,
        'Text_Similarity': text_scores,
        'Price_Similarity': price_scores,
        'Combined_Score': combined_scores,
        'SKU_Match': (dataset['sku'] == master.iloc[matched_indices]['sku'].values).astype(int)
    })
    
    print(f"Matching completed in {time.time() - start_time:.2f} seconds")
    return results

def main():
    input_file = "Product Matching Dataset.xlsx"
    output_file = "Combined_Matching_Results.xlsx"
    #input_test="ur path"
    #dataset_df= load_data(input_test)
    master_df, dataset_df = load_data(input_file)
    # Perform matching with combined score
    results = match_products(master_df, dataset_df)
    
    # Save results to an Excel file
    results.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    print("\nMatching Summary:")
    print(f"Total Items: {len(results):,}")
    print(f"Average Text Similarity: {results['Text_Similarity'].mean():.2f}")
    print(f"Average Price Similarity: {results['Price_Similarity'].mean():.2f}")
    print(f"Average Combined Score: {results['Combined_Score'].mean():.2f}")

if __name__ == "__main__":
    main()
