use tokenizers::{AddedToken, PreTokenizedString, PreTokenizer, SplitDelimiterBehavior};
use tokenizers::models::wordpiece::{WordPiece, WordPieceTrainerBuilder};
use tokenizers::normalizers::BertNormalizer;
use tokenizers::tokenizer::TokenizerBuilder;
use tokenizers::utils::macro_rules_attribute;
use tokenizers::impl_serde_type;
use std::{fs, io};
use std::path::Path;
use clap::Parser;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type !)]
pub struct BertPuncPreTokenizer;

impl PreTokenizer for BertPuncPreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(char::is_whitespace, SplitDelimiterBehavior::Removed))
        // pretokenized.split(|_, s| s.split(is_bert_punc, SplitDelimiterBehavior::Isolated))
    }
}

fn get_files_with_ext(dir: &str, ext: &str) -> io::Result<Vec<String>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some(ext) {
            files.push(String::from(path.to_string_lossy()));
        }
    }
    Ok(files)
}


#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input raw text file (or comma-separated list of files).
    #[arg(long)]
    input_files_dir_path: String,

    /// Input file extension.
    #[arg(long)]
    input_file_ext: String,

    /// Output vocab file path.
    /// The vocabulary file that the BERT model was trained on.
    #[arg(long)]
    output_vocab_file_dir_path: String,

    /// The vocabulary file that the BERT model was trained on.
    #[arg(long)]
    vocab_size: usize,
}


fn main() {
    let args = Args::parse();
    let input_files_dir_path = args.input_files_dir_path;
    let input_file_ext = args.input_file_ext;
    let output_vocab_file_dir_path = args.output_vocab_file_dir_path;
    let vocab_size = args.vocab_size;

    let mut trainer = WordPieceTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .limit_alphabet(1000)
        .continuing_subword_prefix("##".into())
        .special_tokens(vec![
            AddedToken::from(String::from("[PAD]"), true).single_word(true),
            AddedToken::from(String::from("[UNK]"), true).single_word(true),
            AddedToken::from(String::from("[CLS]"), true).single_word(true),
            AddedToken::from(String::from("[SEP]"), true).single_word(true),
            AddedToken::from(String::from("[MASK]"), true).single_word(true),
            AddedToken::from(String::from("[AMOUNT]"), true).single_word(true),
            AddedToken::from(String::from("[ARAB]"), true).single_word(true),
            AddedToken::from(String::from("[ARMN]"), true).single_word(true),
            AddedToken::from(String::from("[BRAI]"), true).single_word(true),
            AddedToken::from(String::from("[CURR]"), true).single_word(true),
            AddedToken::from(String::from("[CYRL]"), true).single_word(true),
            AddedToken::from(String::from("[DATE]"), true).single_word(true),
            AddedToken::from(String::from("[EMAIL]"), true).single_word(true),
            AddedToken::from(String::from("[FOREIGN]"), true).single_word(true),
            AddedToken::from(String::from("[GEOR]"), true).single_word(true),
            AddedToken::from(String::from("[GREK]"), true).single_word(true),
            AddedToken::from(String::from("[HANG]"), true).single_word(true),
            AddedToken::from(String::from("[HANI]"), true).single_word(true),
            AddedToken::from(String::from("[HEBR]"), true).single_word(true),
            AddedToken::from(String::from("[HIND]"), true).single_word(true),
            AddedToken::from(String::from("[ISBN]"), true).single_word(true),
            AddedToken::from(String::from("[JAPN]"), true).single_word(true),
            AddedToken::from(String::from("[THAI]"), true).single_word(true),
            AddedToken::from(String::from("[TIME]"), true).single_word(true),
            AddedToken::from(String::from("[URL]"), true).single_word(true),
            AddedToken::from(String::from("[YEAR]"), true).single_word(true),
        ])
        .build();

    let bert_normalizer = BertNormalizer::new(
        false,
        false,
        Some(false),
        false);

    let wordpiece_decoder = tokenizers::decoders::wordpiece::WordPiece::new(
        String::from("##"),
        false);

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(WordPiece::default())
        .with_normalizer(Some(bert_normalizer))
        .with_post_processor(Some(tokenizers::processors::bert::BertProcessing::default()))
        .with_pre_tokenizer(Some(BertPuncPreTokenizer))
        .with_decoder(Some(wordpiece_decoder))
        .build()
        .unwrap();

    let pretty = true;
    let files = get_files_with_ext(input_files_dir_path.as_str(), input_file_ext.as_str()).unwrap();
    let output_vocab_file_path = Path::join(Path::new(output_vocab_file_dir_path.as_str()), "vocab.json");
    tokenizer.train_from_files(
        &mut trainer,
        files,
    ).unwrap().save(output_vocab_file_path, pretty).unwrap();
}
