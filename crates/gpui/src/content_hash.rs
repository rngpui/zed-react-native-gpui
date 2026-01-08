use crate::{
    AtlasTextureId, Background, BorderStyle, Bounds, BoxShadow, ColorSpace, Corners, Display,
    Edges, Fill, Font, FontFallbacks, FontFeatures, FontStyle, FontWeight, Hsla, ImageId,
    LinearColorStop, ObjectFit, Overflow, Pixels, Point, Rgba, SharedString, Size,
    StrikethroughStyle, TextAlign, TextOverflow, TextRun, UnderlineStyle, Visibility, WhiteSpace,
    Style,
};
use seahash::SeaHasher;
use smallvec::SmallVec;
use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
};

/// A trait for hashing only the content that affects rendering output.
pub trait ContentHash {
    /// Return a stable hash of the content that affects rendering output.
    fn content_hash(&self) -> u64;
}

/// Controls whether primitives can be cached for an element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachePolicy {
    /// Allow primitive caching when a content hash is provided.
    Default,
    /// Disable primitive caching.
    Never,
}

impl CachePolicy {
    /// Returns true when this policy allows cached primitive reuse.
    pub fn allows_caching(self) -> bool {
        matches!(self, CachePolicy::Default)
    }
}

/// A helper for combining multiple content hashes.
#[derive(Default)]
pub struct ContentHasher(SeaHasher);

impl ContentHasher {
    /// Add a u64 to the hash stream.
    pub fn write_u64(&mut self, value: u64) {
        self.0.write_u64(value);
    }

    /// Finish hashing and return the final hash value.
    pub fn finish(self) -> u64 {
        self.0.finish()
    }
}

fn hash_value<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = SeaHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn hash_f32(value: f32) -> u64 {
    value.to_bits() as u64
}

impl ContentHash for bool {
    fn content_hash(&self) -> u64 {
        u64::from(*self)
    }
}

impl ContentHash for u32 {
    fn content_hash(&self) -> u64 {
        *self as u64
    }
}

impl ContentHash for u64 {
    fn content_hash(&self) -> u64 {
        *self
    }
}

impl ContentHash for usize {
    fn content_hash(&self) -> u64 {
        *self as u64
    }
}

impl ContentHash for f32 {
    fn content_hash(&self) -> u64 {
        hash_f32(*self)
    }
}

impl ContentHash for Pixels {
    fn content_hash(&self) -> u64 {
        hash_f32(self.0)
    }
}

impl ContentHash for Point<Pixels> {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        hasher.write_u64(self.x.content_hash());
        hasher.write_u64(self.y.content_hash());
        hasher.finish()
    }
}

impl ContentHash for Size<Pixels> {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        hasher.write_u64(self.width.content_hash());
        hasher.write_u64(self.height.content_hash());
        hasher.finish()
    }
}

impl ContentHash for Bounds<Pixels> {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        hasher.write_u64(self.origin.content_hash());
        hasher.write_u64(self.size.content_hash());
        hasher.finish()
    }
}

impl ContentHash for SharedString {
    fn content_hash(&self) -> u64 {
        hash_value(self.as_ref())
    }
}

impl ContentHash for str {
    fn content_hash(&self) -> u64 {
        hash_value(self)
    }
}

impl ContentHash for String {
    fn content_hash(&self) -> u64 {
        self.as_str().content_hash()
    }
}

impl ContentHash for Hsla {
    fn content_hash(&self) -> u64 {
        hash_value(self)
    }
}

impl ContentHash for Rgba {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        hasher.write_u64(hash_f32(self.r));
        hasher.write_u64(hash_f32(self.g));
        hasher.write_u64(hash_f32(self.b));
        hasher.write_u64(hash_f32(self.a));
        hasher.finish()
    }
}

impl ContentHash for AtlasTextureId {
    fn content_hash(&self) -> u64 {
        hash_value(self)
    }
}

impl ContentHash for ImageId {
    fn content_hash(&self) -> u64 {
        hash_value(self)
    }
}

impl ContentHash for BorderStyle {
    fn content_hash(&self) -> u64 {
        hash_value(self)
    }
}

impl ContentHash for FontWeight {
    fn content_hash(&self) -> u64 {
        hash_value(self)
    }
}

impl ContentHash for FontStyle {
    fn content_hash(&self) -> u64 {
        hash_value(self)
    }
}

impl ContentHash for FontFeatures {
    fn content_hash(&self) -> u64 {
        hash_value(self)
    }
}

impl ContentHash for FontFallbacks {
    fn content_hash(&self) -> u64 {
        hash_value(self)
    }
}

impl ContentHash for Font {
    fn content_hash(&self) -> u64 {
        hash_value(self)
    }
}

impl ContentHash for UnderlineStyle {
    fn content_hash(&self) -> u64 {
        hash_value(self)
    }
}

impl ContentHash for StrikethroughStyle {
    fn content_hash(&self) -> u64 {
        hash_value(self)
    }
}

impl ContentHash for Display {
    fn content_hash(&self) -> u64 {
        match self {
            Display::Block => 0,
            Display::Flex => 1,
            Display::Grid => 2,
            Display::None => 3,
        }
    }
}

impl ContentHash for Visibility {
    fn content_hash(&self) -> u64 {
        match self {
            Visibility::Visible => 0,
            Visibility::Hidden => 1,
        }
    }
}

impl ContentHash for Overflow {
    fn content_hash(&self) -> u64 {
        match self {
            Overflow::Visible => 0,
            Overflow::Clip => 1,
            Overflow::Hidden => 2,
            Overflow::Scroll => 3,
        }
    }
}

impl ContentHash for TextAlign {
    fn content_hash(&self) -> u64 {
        match self {
            TextAlign::Left => 0,
            TextAlign::Center => 1,
            TextAlign::Right => 2,
        }
    }
}

impl ContentHash for ObjectFit {
    fn content_hash(&self) -> u64 {
        match self {
            ObjectFit::Fill => 0,
            ObjectFit::Contain => 1,
            ObjectFit::Cover => 2,
            ObjectFit::ScaleDown => 3,
            ObjectFit::None => 4,
        }
    }
}

impl ContentHash for WhiteSpace {
    fn content_hash(&self) -> u64 {
        match self {
            WhiteSpace::Normal => 0,
            WhiteSpace::Nowrap => 1,
        }
    }
}

impl ContentHash for TextOverflow {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        match self {
            TextOverflow::Truncate(value) => {
                hasher.write_u64(0);
                hasher.write_u64(value.content_hash());
            }
            TextOverflow::TruncateStart(value) => {
                hasher.write_u64(1);
                hasher.write_u64(value.content_hash());
            }
        }
        hasher.finish()
    }
}

impl ContentHash for ColorSpace {
    fn content_hash(&self) -> u64 {
        match self {
            ColorSpace::Srgb => 0,
            ColorSpace::Oklab => 1,
        }
    }
}

impl ContentHash for LinearColorStop {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        hasher.write_u64(self.color.content_hash());
        hasher.write_u64(hash_f32(self.percentage));
        hasher.finish()
    }
}

impl ContentHash for Background {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        match self.tag {
            crate::color::BackgroundTag::Solid => {
                hasher.write_u64(0);
            }
            crate::color::BackgroundTag::LinearGradient => {
                hasher.write_u64(1);
            }
            crate::color::BackgroundTag::PatternSlash => {
                hasher.write_u64(2);
            }
        }
        hasher.write_u64(self.color_space.content_hash());
        hasher.write_u64(self.solid.content_hash());
        hasher.write_u64(hash_f32(self.gradient_angle_or_pattern_height));
        hasher.write_u64(self.stop_count as u64);
        let stop_count = (self.stop_count as usize).min(self.colors.len());
        for stop in &self.colors[..stop_count] {
            hasher.write_u64(stop.content_hash());
        }
        hasher.finish()
    }
}

impl ContentHash for Fill {
    fn content_hash(&self) -> u64 {
        match self {
            Fill::Color(background) => background.content_hash(),
        }
    }
}

impl ContentHash for BoxShadow {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        hasher.write_u64(self.color.content_hash());
        hasher.write_u64(self.offset.content_hash());
        hasher.write_u64(self.blur_radius.content_hash());
        hasher.write_u64(self.spread_radius.content_hash());
        hasher.finish()
    }
}

impl ContentHash for TextRun {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        hasher.write_u64(self.len.content_hash());
        hasher.write_u64(self.font.content_hash());
        hasher.write_u64(self.color.content_hash());
        hasher.write_u64(self.background_color.content_hash());
        hasher.write_u64(self.underline.content_hash());
        hasher.write_u64(self.strikethrough.content_hash());
        hasher.finish()
    }
}

/// Hash the visual style properties that affect element rendering.
pub fn style_content_hash(style: &Style, rem_size: Pixels) -> u64 {
    let mut hasher = ContentHasher::default();
    hasher.write_u64(style.display.content_hash());
    hasher.write_u64(style.visibility.content_hash());
    hasher.write_u64(style.overflow.x.content_hash());
    hasher.write_u64(style.overflow.y.content_hash());
    hasher.write_u64(style.background.content_hash());
    hasher.write_u64(style.border_color.content_hash());
    hasher.write_u64(style.border_style.content_hash());
    hasher.write_u64(style.border_widths.to_pixels(rem_size).content_hash());
    hasher.write_u64(style.corner_radii.to_pixels(rem_size).content_hash());
    hasher.write_u64(style.box_shadow.content_hash());
    hasher.write_u64(style.opacity.content_hash());
    hasher.finish()
}

impl<T> ContentHash for Edges<T>
where
    T: ContentHash + Clone + Debug + Default + PartialEq,
{
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        hasher.write_u64(self.top.content_hash());
        hasher.write_u64(self.right.content_hash());
        hasher.write_u64(self.bottom.content_hash());
        hasher.write_u64(self.left.content_hash());
        hasher.finish()
    }
}

impl<T> ContentHash for Corners<T>
where
    T: ContentHash + Clone + Debug + Default + PartialEq,
{
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        hasher.write_u64(self.top_left.content_hash());
        hasher.write_u64(self.top_right.content_hash());
        hasher.write_u64(self.bottom_right.content_hash());
        hasher.write_u64(self.bottom_left.content_hash());
        hasher.finish()
    }
}

impl<T: ContentHash> ContentHash for Option<T> {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        match self {
            None => hasher.write_u64(0),
            Some(value) => {
                hasher.write_u64(1);
                hasher.write_u64(value.content_hash());
            }
        }
        hasher.finish()
    }
}

impl<T: ContentHash> ContentHash for Vec<T> {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        hasher.write_u64(self.len() as u64);
        for value in self {
            hasher.write_u64(value.content_hash());
        }
        hasher.finish()
    }
}

impl<T: ContentHash, const N: usize> ContentHash for SmallVec<[T; N]> {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        hasher.write_u64(self.len() as u64);
        for value in self {
            hasher.write_u64(value.content_hash());
        }
        hasher.finish()
    }
}
