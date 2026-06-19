use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Attribute, Data, DeriveInput, Fields, LitStr, Result, parse_macro_input};

#[proc_macro_derive(LLMActions, attributes(llm))]
pub fn derive_llm_actions(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_llm_actions(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_llm_actions(input: DeriveInput) -> Result<proc_macro2::TokenStream> {
    let enum_ident = input.ident;

    let Data::Enum(data_enum) = input.data else {
        return Err(syn::Error::new_spanned(
            enum_ident.clone(),
            "LLMActions can only be derived for enums",
        ));
    };

    let mut variant_tokens = Vec::new();
    let mut tool_name_literals = Vec::new();
    let mut tool_definition_exprs = Vec::new();

    for variant in data_enum.variants {
        let variant_ident = variant.ident;
        let variant_name = variant_ident.to_string();
        let tool_name = llm_name(&variant.attrs)
            .unwrap_or_else(|| LitStr::new(&to_snake_case(&variant_name), variant_ident.span()));
        let tool_description = llm_description(&variant.attrs).unwrap_or_else(|| {
            LitStr::new(
                &format!("{} tool", humanize_ident(&variant_name)),
                variant_ident.span(),
            )
        });

        let args_struct_ident =
            format_ident!("__llm_actions_{}_{}_args", enum_ident, variant_ident);

        let variant_span = variant_ident.span();

        let (field_defs, _field_idents, _field_value_tokens) = match variant.fields {
            Fields::Named(fields_named) => {
                let mut defs = Vec::new();
                let mut idents = Vec::new();
                let mut values = Vec::new();

                for field in fields_named.named {
                    let field_ident = field.ident.expect("named field");
                    let field_ty = field.ty;
                    let field_name = field_ident.to_string();
                    let field_description = llm_description(&field.attrs).unwrap_or_else(|| {
                        LitStr::new(&humanize_ident(&field_name), field_ident.span())
                    });
                    let field_rename = llm_rename(&field.attrs)
                        .unwrap_or_else(|| LitStr::new(&field_name, field_ident.span()));

                    defs.push(quote! {
                        #[serde(rename = #field_rename)]
                        #[schemars(description = #field_description)]
                        pub #field_ident: #field_ty,
                    });
                    idents.push(field_ident.clone());
                    values.push(quote! { #field_ident: args.#field_ident });
                }

                (defs, idents, values)
            }
            Fields::Unit => (Vec::new(), Vec::new(), Vec::new()),
            Fields::Unnamed(_) => {
                return Err(syn::Error::new(
                    variant_span,
                    "LLMActions currently supports only unit and named-field variants",
                ));
            }
        };

        let args_struct = if field_defs.is_empty() {
            quote! {
                #[allow(non_camel_case_types)]
                #[derive(Debug, Clone, ::serde::Serialize, ::serde::Deserialize, ::schemars::JsonSchema)]
                pub struct #args_struct_ident;
            }
        } else {
            quote! {
                #[allow(non_camel_case_types)]
                #[derive(Debug, Clone, ::serde::Serialize, ::serde::Deserialize, ::schemars::JsonSchema)]
                pub struct #args_struct_ident {
                    #(#field_defs)*
                }
            }
        };

        variant_tokens.push(quote! {
            #args_struct
        });

        tool_name_literals.push(tool_name.clone());
        tool_definition_exprs.push(quote! {
            ::bevellm::LlmToolDefinition::new(
                #tool_name,
                #tool_description,
                ::serde_json::to_value(::schemars::schema_for!(#args_struct_ident))
                    .expect("LLMActions tool schemas must serialize"),
            )
        });
    }

    Ok(quote! {
        impl #enum_ident {
            pub fn llm_tool_names() -> &'static [&'static str] {
                &[#(#tool_name_literals),*]
            }

            pub fn llm_tool_definitions() -> ::std::vec::Vec<::bevellm::LlmToolDefinition> {
                vec![#(#tool_definition_exprs),*]
            }

            pub fn llm_tool_set(
                sender: ::std::sync::mpsc::Sender<Self>,
            ) -> ::bevellm::LlmToolSet {
                let _ = sender;
                ::bevellm::LlmToolSet::new(
                    vec![
                        #(
                            ::std::string::ToString::to_string(#tool_name_literals)
                        ),*
                    ],
                    vec![#(#tool_definition_exprs),*],
                )
            }
        }

        #(#variant_tokens)*
    })
}

fn llm_name(attrs: &[Attribute]) -> Option<LitStr> {
    llm_named_string(attrs, "name")
}

fn llm_description(attrs: &[Attribute]) -> Option<LitStr> {
    llm_named_string(attrs, "description").or_else(|| doc_string(attrs))
}

fn llm_rename(attrs: &[Attribute]) -> Option<LitStr> {
    llm_named_string(attrs, "rename")
}

fn llm_named_string(attrs: &[Attribute], key: &str) -> Option<LitStr> {
    for attr in attrs {
        if !attr.path().is_ident("llm") {
            continue;
        }

        let mut value = None;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident(key) {
                let lit: LitStr = meta.value()?.parse()?;
                value = Some(lit);
            }
            Ok(())
        });

        if value.is_some() {
            return value;
        }
    }

    None
}

fn doc_string(attrs: &[Attribute]) -> Option<LitStr> {
    let mut docs = Vec::new();

    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }

        let syn::Meta::NameValue(meta) = &attr.meta else {
            continue;
        };

        let syn::Expr::Lit(expr_lit) = &meta.value else {
            continue;
        };

        let syn::Lit::Str(lit) = &expr_lit.lit else {
            continue;
        };

        let line = lit.value().trim().to_string();
        if !line.is_empty() {
            docs.push(line);
        }
    }

    if docs.is_empty() {
        None
    } else {
        Some(LitStr::new(&docs.join(" "), proc_macro2::Span::call_site()))
    }
}

fn humanize_ident(name: &str) -> String {
    let mut out = String::new();
    let mut prev_lower_or_digit = false;

    for ch in name.chars() {
        if ch == '_' {
            out.push(' ');
            prev_lower_or_digit = false;
            continue;
        }

        if ch.is_uppercase() {
            if prev_lower_or_digit && !out.ends_with(' ') {
                out.push(' ');
            }
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            prev_lower_or_digit = false;
        } else {
            out.push(ch);
            prev_lower_or_digit = ch.is_lowercase() || ch.is_numeric();
        }
    }

    out.trim().to_string()
}

fn to_snake_case(name: &str) -> String {
    let mut out = String::new();
    let mut chars = name.chars().peekable();
    let mut prev_is_lower_or_digit = false;

    while let Some(ch) = chars.next() {
        if ch == '_' {
            if !out.ends_with('_') {
                out.push('_');
            }
            prev_is_lower_or_digit = false;
            continue;
        }

        if ch.is_uppercase() {
            let next_is_lower = chars
                .peek()
                .map(|next| next.is_lowercase())
                .unwrap_or(false);
            if !out.is_empty() && (prev_is_lower_or_digit || next_is_lower) && !out.ends_with('_') {
                out.push('_');
            }
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            prev_is_lower_or_digit = false;
        } else {
            out.push(ch);
            prev_is_lower_or_digit = ch.is_lowercase() || ch.is_numeric();
        }
    }

    out.trim_matches('_').to_string()
}
