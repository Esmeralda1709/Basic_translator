from src.search import top_k_translation, print_results
from src.load_model import load_model, print_model_info



print("\n=========== Traducotr del Ingles al Español ===========")

(vocab_en, vocab_es, reverse_vocab_es,
 emb_en, emb_es, W ) = load_model( "models" )


print_model_info(vocab_en, vocab_es, emb_en, emb_es, W)


print("\nEscribe '0' para finalizar.")

while True:
    word = input("\nPalabra en ingles: ")
    word = word.strip().lower()

    if word == "0":
        print("\nFinalizando traductor")
        break

    results = top_k_translation(word, vocab_en, reverse_vocab_es,
                                emb_en, emb_es, W, 5)

    print_results(results)