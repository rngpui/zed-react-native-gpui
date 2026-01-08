use std::{fs, path::Path, sync::Arc};

use crate::{
    App, Asset, Bounds, ContentHash, ContentHasher, Element, GlobalElementId, Hitbox,
    InspectorElementId, InteractiveElement, Interactivity, IntoElement, LayoutId, Pixels, Point,
    Radians, SharedString, Size, StyleRefinement, Styled, TransformationMatrix, Window, point, px,
    radians, size,
};
use util::ResultExt;

/// An SVG element.
pub struct Svg {
    interactivity: Interactivity,
    transformation: Option<Transformation>,
    path: Option<SharedString>,
    external_path: Option<SharedString>,
}

/// Create a new SVG element.
#[track_caller]
pub fn svg() -> Svg {
    Svg {
        interactivity: Interactivity::new(),
        transformation: None,
        path: None,
        external_path: None,
    }
}

impl Svg {
    /// Set the path to the SVG file for this element.
    pub fn path(mut self, path: impl Into<SharedString>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set the path to the SVG file for this element.
    pub fn external_path(mut self, path: impl Into<SharedString>) -> Self {
        self.external_path = Some(path.into());
        self
    }

    /// Transform the SVG element with the given transformation.
    /// Note that this won't effect the hitbox or layout of the element, only the rendering.
    pub fn with_transformation(mut self, transformation: Transformation) -> Self {
        self.transformation = Some(transformation);
        self
    }
}

impl Element for Svg {
    type RequestLayoutState = ();
    type PrepaintState = Option<Hitbox>;

    fn id(&self) -> Option<crate::ElementId> {
        self.interactivity.element_id.clone()
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        self.interactivity.source_location()
    }

    fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let layout_id = self.interactivity.request_layout(
            global_id,
            inspector_id,
            window,
            cx,
            |style, window, cx| window.request_layout(style, None, cx),
        );
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Hitbox> {
        let base_hash = self.interactivity.content_hash;
        let path_hash = self.path.content_hash();
        let external_path_hash = self.external_path.content_hash();
        let transform_hash = self.transformation.content_hash();
        let mut content_hash = None;
        let hitbox = self.interactivity.prepaint(
            global_id,
            inspector_id,
            bounds,
            bounds.size,
            window,
            cx,
            |style, _, hitbox, _window, _cx| {
                let mut hasher = ContentHasher::default();
                if let Some(base) = base_hash {
                    hasher.write_u64(base);
                }
                hasher.write_u64(path_hash);
                hasher.write_u64(external_path_hash);
                hasher.write_u64(style.text.color.content_hash());
                hasher.write_u64(transform_hash);
                content_hash = Some(hasher.finish());
                hitbox
            },
        );
        self.interactivity.content_hash = content_hash;
        hitbox
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        hitbox: &mut Option<Hitbox>,
        window: &mut Window,
        cx: &mut App,
    ) where
        Self: Sized,
    {
        self.interactivity.paint(
            global_id,
            inspector_id,
            bounds,
            hitbox.as_ref(),
            window,
            cx,
            |style, window, cx| {
                if let Some((path, color)) = self.path.as_ref().zip(style.text.color) {
                    let transformation = self
                        .transformation
                        .as_ref()
                        .map(|transformation| {
                            transformation.into_matrix(bounds.center(), window.scale_factor())
                        })
                        .unwrap_or_default();

                    window
                        .paint_svg(bounds, path.clone(), None, transformation, color, cx)
                        .log_err();
                } else if let Some((path, color)) =
                    self.external_path.as_ref().zip(style.text.color)
                {
                    let Some(bytes) = window
                        .use_asset::<SvgAsset>(path, cx)
                        .and_then(|asset| asset.log_err())
                    else {
                        return;
                    };

                    let transformation = self
                        .transformation
                        .as_ref()
                        .map(|transformation| {
                            transformation.into_matrix(bounds.center(), window.scale_factor())
                        })
                        .unwrap_or_default();

                    window
                        .paint_svg(
                            bounds,
                            path.clone(),
                            Some(&bytes),
                            transformation,
                            color,
                            cx,
                        )
                        .log_err();
                }
            },
        )
    }

    fn content_hash(
        &self,
        _id: Option<&GlobalElementId>,
        _bounds: Bounds<Pixels>,
        _window: &Window,
        _cx: &App,
    ) -> Option<u64> {
        self.interactivity.content_hash
    }
}

impl IntoElement for Svg {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Styled for Svg {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.interactivity.base_style
    }
}

impl InteractiveElement for Svg {
    fn interactivity(&mut self) -> &mut Interactivity {
        &mut self.interactivity
    }
}

/// A transformation to apply to an SVG element.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transformation {
    scale: Size<f32>,
    translate: Point<Pixels>,
    rotate: Radians,
}

impl Default for Transformation {
    fn default() -> Self {
        Self {
            scale: size(1.0, 1.0),
            translate: point(px(0.0), px(0.0)),
            rotate: radians(0.0),
        }
    }
}

impl Transformation {
    /// Create a new Transformation with the specified scale along each axis.
    pub fn scale(scale: Size<f32>) -> Self {
        Self {
            scale,
            translate: point(px(0.0), px(0.0)),
            rotate: radians(0.0),
        }
    }

    /// Create a new Transformation with the specified translation.
    pub fn translate(translate: Point<Pixels>) -> Self {
        Self {
            scale: size(1.0, 1.0),
            translate,
            rotate: radians(0.0),
        }
    }

    /// Create a new Transformation with the specified rotation in radians.
    pub fn rotate(rotate: impl Into<Radians>) -> Self {
        let rotate = rotate.into();
        Self {
            scale: size(1.0, 1.0),
            translate: point(px(0.0), px(0.0)),
            rotate,
        }
    }

    /// Update the scaling factor of this transformation.
    pub fn with_scaling(mut self, scale: Size<f32>) -> Self {
        self.scale = scale;
        self
    }

    /// Update the translation value of this transformation.
    pub fn with_translation(mut self, translate: Point<Pixels>) -> Self {
        self.translate = translate;
        self
    }

    /// Update the rotation angle of this transformation.
    pub fn with_rotation(mut self, rotate: impl Into<Radians>) -> Self {
        self.rotate = rotate.into();
        self
    }

    fn into_matrix(self, center: Point<Pixels>, scale_factor: f32) -> TransformationMatrix {
        // MonochromeSprite bounds are in device (ScaledPixels) space, so the transform
        // must also be in device space. Scale the translation values accordingly.
        //Note: if you read this as a sequence of matrix multiplications, start from the bottom
        let scaled_center = point(
            px(center.x.0 * scale_factor),
            px(center.y.0 * scale_factor),
        );
        let scaled_translate = point(
            px(self.translate.x.0 * scale_factor),
            px(self.translate.y.0 * scale_factor),
        );
        TransformationMatrix::unit()
            .translate(scaled_center + scaled_translate)
            .rotate(self.rotate)
            .scale(self.scale)
            .translate(point(px(-scaled_center.x.0), px(-scaled_center.y.0)))
    }
}

impl ContentHash for Transformation {
    fn content_hash(&self) -> u64 {
        let mut hasher = ContentHasher::default();
        hasher.write_u64(self.scale.width.to_bits() as u64);
        hasher.write_u64(self.scale.height.to_bits() as u64);
        hasher.write_u64(self.translate.x.content_hash());
        hasher.write_u64(self.translate.y.content_hash());
        hasher.write_u64(self.rotate.0.to_bits() as u64);
        hasher.finish()
    }
}

enum SvgAsset {}

impl Asset for SvgAsset {
    type Source = SharedString;
    type Output = Result<Arc<[u8]>, Arc<std::io::Error>>;

    fn load(
        source: Self::Source,
        _cx: &mut App,
    ) -> impl Future<Output = Self::Output> + Send + 'static {
        async move {
            let bytes = fs::read(Path::new(source.as_ref())).map_err(|e| Arc::new(e))?;
            let bytes = Arc::from(bytes);
            Ok(bytes)
        }
    }
}
