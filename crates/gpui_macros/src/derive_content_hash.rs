use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, GenericParam, Index, parse_macro_input, parse_quote};

pub fn derive_content_hash(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let mut generics = input.generics;
    for param in &mut generics.params {
        if let GenericParam::Type(type_param) = param {
            type_param.bounds.push(parse_quote!(::gpui::ContentHash));
        }
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let body = match input.data {
        Data::Struct(data) => {
            let field_hashes = match data.fields {
                Fields::Named(fields) => fields
                    .named
                    .iter()
                    .map(|field| {
                        let ident = field.ident.as_ref().expect("named field");
                        quote! {
                            hasher.write_u64(::gpui::ContentHash::content_hash(&self.#ident));
                        }
                    })
                    .collect::<Vec<_>>(),
                Fields::Unnamed(fields) => fields
                    .unnamed
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        let index = Index::from(index);
                        quote! {
                            hasher.write_u64(::gpui::ContentHash::content_hash(&self.#index));
                        }
                    })
                    .collect::<Vec<_>>(),
                Fields::Unit => Vec::new(),
            };

            quote! {
                let mut hasher = ::gpui::content_hash::ContentHasher::default();
                #(#field_hashes)*
                hasher.finish()
            }
        }
        Data::Enum(data) => {
            let variant_arms = data
                .variants
                .iter()
                .enumerate()
                .map(|(variant_index, variant)| {
                    let variant_ident = &variant.ident;
                    let variant_index = variant_index as u64;

                    match &variant.fields {
                        Fields::Named(fields) => {
                            let bindings = fields.named.iter().map(|field| {
                                field.ident.as_ref().expect("named field")
                            });
                            let hash_calls = fields.named.iter().map(|field| {
                                let ident = field.ident.as_ref().expect("named field");
                                quote! {
                                    hasher.write_u64(::gpui::ContentHash::content_hash(#ident));
                                }
                            });
                            quote! {
                                Self::#variant_ident { #( ref #bindings ),* } => {
                                    let mut hasher = ::gpui::content_hash::ContentHasher::default();
                                    hasher.write_u64(#variant_index);
                                    #(#hash_calls)*
                                    hasher.finish()
                                }
                            }
                        }
                        Fields::Unnamed(fields) => {
                            let bindings = fields.unnamed.iter().enumerate().map(|(i, _)| {
                                syn::Ident::new(&format!("field_{i}"), proc_macro2::Span::call_site())
                            });
                            let hash_calls = fields.unnamed.iter().enumerate().map(|(i, _)| {
                                let ident = syn::Ident::new(
                                    &format!("field_{i}"),
                                    proc_macro2::Span::call_site(),
                                );
                                quote! {
                                    hasher.write_u64(::gpui::ContentHash::content_hash(#ident));
                                }
                            });
                            quote! {
                                Self::#variant_ident( #( ref #bindings ),* ) => {
                                    let mut hasher = ::gpui::content_hash::ContentHasher::default();
                                    hasher.write_u64(#variant_index);
                                    #(#hash_calls)*
                                    hasher.finish()
                                }
                            }
                        }
                        Fields::Unit => quote! {
                            Self::#variant_ident => {
                                let mut hasher = ::gpui::content_hash::ContentHasher::default();
                                hasher.write_u64(#variant_index);
                                hasher.finish()
                            }
                        },
                    }
                })
                .collect::<Vec<_>>();

            quote! {
                match self {
                    #(#variant_arms)*
                }
            }
        }
        Data::Union(_) => {
            return syn::Error::new_spanned(
                name,
                "#[derive(ContentHash)] can only be used with structs or enums",
            )
            .to_compile_error()
            .into();
        }
    };

    quote! {
        impl #impl_generics ::gpui::ContentHash for #name #ty_generics #where_clause {
            fn content_hash(&self) -> u64 {
                #body
            }
        }
    }
    .into()
}
